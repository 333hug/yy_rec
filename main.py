import os, shutil, uuid, json, torch, psutil, subprocess, threading, queue
os.environ["MODELSCOPE_CACHE"] = "/home/ubuntu/gjl/yy_rec/model"
import numpy as np
import faiss
import soundfile as sf
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, LargeBinary, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy.orm.attributes import flag_modified

# ===== 🟢 模型导入 =====
from funasr import AutoModel
from openai import OpenAI

# ==================== 1. 全局配置与安全 ====================
os.makedirs("static/uploads", exist_ok=True)
DB_URL = "sqlite:///./voice_system.db"
FAISS_INDEX_PATH = "speaker.index"

SECRET_KEY = "your_super_secret_key_here"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

# ==================== 2. 数据库与 FAISS ====================
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class AudioTask(Base):
    __tablename__ = "audio_tasks"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    file_path = Column(String)
    status = Column(String, default="pending") 
    result_json = Column(JSON, nullable=True)   
    summary = Column(String, nullable=True) 
    created_at = Column(DateTime, default=datetime.utcnow)

class SpeakerProfile(Base):
    __tablename__ = "speaker_profiles"
    id = Column(Integer, primary_key=True) 
    name = Column(String)
    embedding = Column(LargeBinary)

Base.metadata.create_all(bind=engine)

DIMENSION = 192
if os.path.exists(FAISS_INDEX_PATH):
    speaker_index = faiss.read_index(FAISS_INDEX_PATH)
else:
    speaker_index = faiss.IndexFlatIP(DIMENSION)

print("🚀 正在全局加载模型到显存...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("⏳ [1/2] 正在加载 阿里 FunASR 全家桶...")
asr_model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++", device=str(device), disable_update=True)
print("⏳ [2/2] 正在加载 阿里 CAM++...")
emb_model = AutoModel(model="cam++", device=str(device), disable_update=True)
print("✅ 所有阿里顶尖模型加载完毕！服务已就绪。")


# ==================== 4. 🟢 后台异步 Worker 线程 ====================
task_queue = queue.Queue()

def audio_worker():
    print("🚀 后台推理 Worker 已启动，等待任务排队...")
    while True:
        task_id = task_queue.get()
        if task_id is None: break
        
        db = SessionLocal()
        std_file_path = None
        task = None
        try:
            task = db.query(AudioTask).filter(AudioTask.id == task_id).first()
            if not task: continue

            # 通知前端任务开始
            task.status = "processing"
            db.commit()

            print(f"\n=====================================")
            print(f"⚙️ [后台启动] 开始处理任务: {task.filename}")
            
            # [0/3] FFmpeg 标准化音频
            raw_file_path = task.file_path
            std_file_path = raw_file_path + "_std.wav"
            print("⏳ [0/3] FFmpeg 正在标准化音频 (16kHz 单声道)...")
            subprocess.run(["ffmpeg", "-y", "-i", raw_file_path, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", std_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            file_path = std_file_path 

            # [1/3] 阿里 FunASR 端到端
            print("⏳ [1/3] FunASR 端到端全链路分析中 (速度极快)...")
            res = asr_model.generate(input=file_path, batch_size_s=300, sentence_timestamp=True)
            sentence_info = res[0]["sentence_info"] if len(res) > 0 and "sentence_info" in res[0] else[]
            print(f"✅ 提取出 {len(sentence_info)} 句文本，且已全部分配好发言人！")

            # [2/3] 按发言人提取高精声纹特征
            print("⏳ [2/3] 正在提取音频特征...")
            wav_data, sr = sf.read(file_path, dtype='float32') 
            spk_to_identity = {}
            for s_dict in sentence_info:
                spk = s_dict.get("spk", -1) 
                start_ms, end_ms, s_text = s_dict["start"], s_dict["end"], s_dict.get("text", "").strip()
                if not s_text: continue
                if spk not in spk_to_identity: spk_to_identity[spk] = {"audio_chunks":[]}
                start_idx, end_idx = max(0, int((start_ms / 1000.0) * sr)), max(0, int((end_ms / 1000.0) * sr))
                spk_to_identity[spk]["audio_chunks"].append(wav_data[start_idx:end_idx])

            # [3/3] 计算特征并匹配 FAISS
            print("⏳ [3/3] 计算平均声纹特征并匹配数据库...")
            speaker_counter = 1
            for spk, data in spk_to_identity.items():
                if not data["audio_chunks"]: continue
                concat_audio = np.concatenate(data["audio_chunks"])
                emb_res = emb_model.generate(input=concat_audio)
                if len(emb_res) == 0 or 'spk_embedding' not in emb_res[0]:
                    data["name"], data["emb_hex"] = "未知", ""
                    continue
                    
                emb = np.array(emb_res[0]['spk_embedding']).flatten()
                emb = emb / np.linalg.norm(emb)
                emb_query = emb.reshape(1, -1).astype('float32')
                
                name = f"陌生人 {speaker_counter}"
                if speaker_index.ntotal > 0:
                    D, I = speaker_index.search(emb_query, 1)
                    if D[0][0] > 0.65:
                        match_spk = db.query(SpeakerProfile).filter(SpeakerProfile.id == int(I[0][0])).first()
                        if match_spk: name = match_spk.name
                        
                if "陌生人" in name: speaker_counter += 1
                data["name"], data["emb_hex"] = name, emb.tobytes().hex()

            # 重组时间线聊天记录
            print("⏳ 正在按时间线重组剧本式聊天流...")
            timeline_dialogue, current_block =[], None
            for s_dict in sentence_info:
                spk, s_text = s_dict.get("spk", -1), s_dict.get("text", "").strip()
                s_start, s_end = s_dict["start"] / 1000.0, s_dict["end"] / 1000.0
                if not s_text or spk not in spk_to_identity: continue
                identity = spk_to_identity[spk]

                if current_block is None or current_block["spk"] != spk:
                    if current_block is not None: timeline_dialogue.append(current_block)
                    current_block = {"spk": spk, "name": identity.get("name", "未知"), "emb_hex": identity.get("emb_hex", ""), "start": s_start, "end": s_end, "texts":[s_text]}
                else:
                    current_block["end"] = s_end
                    current_block["texts"].append(s_text)
                    
            if current_block is not None: timeline_dialogue.append(current_block)

            final_json =[{"id": idx, "start": round(b["start"], 2), "end": round(b["end"], 2), "name": b["name"], "text": " ".join(b["texts"]), "emb_hex": b["emb_hex"]} for idx, b in enumerate(timeline_dialogue)]

            task.result_json = final_json
            task.status = "completed"
            db.commit()
            print(f"✅ 任务 {task.id} 处理完毕！结果已落库。")
            print("=====================================\n")

        except Exception as e:
            print(f"\n❌ 任务 {task_id} 执行失败: {e}")
            if task:
                task.status = "failed"
                db.commit()
        finally:
            if std_file_path and os.path.exists(std_file_path):
                os.remove(std_file_path)
            db.close()
            task_queue.task_done()

# 启动守护线程跑后台推理任务
threading.Thread(target=audio_worker, daemon=True).start()

# ==================== 5. 依赖注入与 API 路由 ====================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise HTTPException(status_code=401)
    except JWTError: raise HTTPException(status_code=401)
    user = db.query(User).filter(User.username == username).first()
    if user is None: raise HTTPException(status_code=401)
    return user

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("templates/index.html")

@app.post("/api/register")
def register(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == form_data.username).first(): raise HTTPException(status_code=400, detail="用户名已存在")
    hashed_password = pwd_context.hash(form_data.password)
    db.add(User(username=form_data.username, password_hash=hashed_password))
    db.commit()
    return {"msg": "注册成功，请登录"}

@app.post("/api/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.password_hash): raise HTTPException(status_code=400, detail="用户名或密码错误")
    access_token = jwt.encode({"sub": user.username, "exp": datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": access_token, "token_type": "bearer", "username": user.username}

@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # 🟢 恢复异步：这里只做接收并保存文件，处理逻辑完全交给后台线程
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')): raise HTTPException(status_code=400, detail="仅支持音频格式")
    
    task_uuid = uuid.uuid4().hex
    raw_file_path = f"static/uploads/{task_uuid}_{file.filename}"
    with open(raw_file_path, "wb") as f: shutil.copyfileobj(file.file, f)
    
    task = AudioTask(user_id=user.id, filename=file.filename, file_path=raw_file_path, status="pending")
    db.add(task)
    db.commit()
    db.refresh(task)

    # 将任务送入后台队列，然后瞬间返回成功
    task_queue.put(task.id)
    return {"msg": "文件上传成功，已加入后台识别队列", "task_id": task.id}

@app.get("/api/tasks")
def list_tasks(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(AudioTask).filter(AudioTask.user_id == user.id).order_by(AudioTask.created_at.desc()).all()

@app.post("/api/label")
async def label_speaker(name: str = Form(...), emb_hex: str = Form(...), task_id: int = Form(...), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    global speaker_index
    new_emb = np.frombuffer(bytes.fromhex(emb_hex), dtype='float32').reshape(1, -1)
    new_emb = new_emb / np.linalg.norm(new_emb)
    
    existing_profile = db.query(SpeakerProfile).filter(SpeakerProfile.name == name).first()
    if existing_profile:
        old_emb = np.frombuffer(existing_profile.embedding, dtype='float32').reshape(1, -1)
        merged_emb = np.mean([old_emb[0], new_emb[0]], axis=0).astype('float32').reshape(1, -1)
        existing_profile.embedding = (merged_emb / np.linalg.norm(merged_emb)).tobytes()
        db.add(existing_profile)
    else:
        db.add(SpeakerProfile(id=db.query(SpeakerProfile).count(), name=name, embedding=new_emb.tobytes()))
        
    db.commit()

    speaker_index = faiss.IndexFlatIP(DIMENSION)
    for p in db.query(SpeakerProfile).order_by(SpeakerProfile.id).all():
        speaker_index.add(np.frombuffer(p.embedding, dtype='float32').reshape(1, -1))
    faiss.write_index(speaker_index, FAISS_INDEX_PATH)
    
    task = db.query(AudioTask).filter(AudioTask.id == task_id, AudioTask.user_id == user.id).first()
    if task and task.result_json:
        updated_json =[dict(b) | {"name": name} if dict(b).get("emb_hex") == emb_hex else dict(b) for b in task.result_json]
        task.result_json = updated_json
        flag_modified(task, "result_json") 
        db.add(task)
        db.commit()

    return {"msg": "更新成功"}

@app.post("/api/update_text")
async def update_text(task_id: int = Form(...), block_id: int = Form(...), new_text: str = Form(...), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    task = db.query(AudioTask).filter(AudioTask.id == task_id, AudioTask.user_id == user.id).first()
    if task and task.result_json:
        updated_json =[]
        for block in task.result_json:
            new_block = dict(block)
            if new_block.get("id") == block_id:
                new_block["text"] = new_text
            updated_json.append(new_block)
        task.result_json = updated_json
        flag_modified(task, "result_json")
        db.commit()
    return {"msg": "修改成功"}

@app.post("/api/generate_summary")
def generate_summary(
    task_id: int = Form(...), 
    api_key: str = Form(...),        # 🟢 接收前端传来的 API Key
    base_url: str = Form(...),       # 🟢 接收前端传来的 URL
    model_name: str = Form(...),     # 🟢 接收前端传来的模型名
    system_prompt: str = Form(...),  # 🟢 接收前端传来的提示词
    db: Session = Depends(get_db), 
    user: User = Depends(get_current_user)
):
    task = db.query(AudioTask).filter(AudioTask.id == task_id, AudioTask.user_id == user.id).first()
    if not task or not task.result_json: 
        raise HTTPException(status_code=400, detail="记录不存在")

    dialogue_str = "\n".join([f"{b['name']}: {b['text']}" for b in task.result_json])
    
    try:
        # 🟢 动态实例化大模型客户端，支持任何兼容 OpenAI 格式的模型
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dialogue_str}
            ]
        )
        summary_text = response.choices[0].message.content
        task.summary = summary_text
        db.commit()
        return {"summary": summary_text}
    except Exception as e:
        print(f"❌ 大模型调用报错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"大模型配置有误或调用失败: {str(e)}")

@app.get("/api/monitor")
def get_monitor():
    return { 
        "cpu": psutil.cpu_percent(interval=None), 
        "ram": psutil.virtual_memory().percent, 
        "queue_size": task_queue.qsize()  # 
    }