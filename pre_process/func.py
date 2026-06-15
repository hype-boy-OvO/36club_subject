import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

class Categorized_Dataset(Dataset):
    def __init__(self, excel_path, target_category=None, model_name="kykim/bert-kor-base", max_len=128,tokenizer=None):
        """
        Args:
            excel_path (str): 엑셀 파일 경로
            target_category (str): 뽑아오고 싶은 데이터 유형 
                                   ('유행어/신조어', 'IT 신기술', '생물/바이오 분야', '일반 일상 문장')
                                   
        """
        # 1. 엑셀 파일 로드 (상단 타이틀 3줄 건너뛰기)
        df = pd.read_excel(excel_path, skiprows=3)
        
        # 2. 지정한 데이터 유형(대분류)만 필터링하기
        if target_category is not None:
            # 엑셀의 2번째 컬럼(대분류) 기준 필터링
            df = df[df.iloc[:, 2] == target_category]
            
        # 4번째 컬럼(문장 리스트)에서 텍스트 추출
        self.sentences = df.iloc[:, 4].dropna().astype(str).tolist()
        
        # 토크나이저 및 옵션 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # 마스킹을 하지 않은 깨끗한 서브워드 토큰 상태로 반환
        tokens = self.tokenizer(
            self.sentences[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        }
    
def get_dataloader(excel_path, target_category, batch_size=8, max_len=128, mlm_probability=0.15, model_name="kykim/bert-kor-base", shuffle=True, if_train=0):
    """
    엑셀 파일과 카테고리를 주면 학습 준비가 완료된 DataLoader를 즉시 반환합니다.
    """
    # 1. 내부에서 토크나이저 한 번만 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. 데이터셋 생성 (토크나이저 주입)
    dataset = Categorized_Dataset(
        excel_path=excel_path, 
        target_category=target_category, 
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    # 3. 내부에서 데이터 콜레이터 정의
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )
    
    # 4. 최종 데이터로더 생성 및 반환
    if target_category == "일반 일상 문장" and not if_train:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
        )
    else:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=data_collator
        )
    
    return loader