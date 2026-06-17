import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

class Custom_Sentence_Dataset(Dataset):
    def __init__(self, sentences, model_name="kykim/bert-kor-base", max_len=128, tokenizer=None):
        """
        Args:
            sentences (list): 학습/테스트에 사용할 문장 리스트 (예: ['문장1', '문장2'])
            model_name (str): 사용할 BERT 모델명
            max_len (int): 토큰 최대 길이
            tokenizer: 외부에서 전달받은 토크나이저 (없으면 내부에서 로드)
        """
        # 입력받은 문장 리스트에서 결측치 제거 및 문자열 변환
        self.sentences = [str(s) for s in sentences if pd.notna(s)]
        
        # 외부 토크나이저가 있으면 사용하고, 없으면 새로 로드
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
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
    
def get_custom_dataloader(sentences, batch_size=1, max_len=128, mlm_probability=0.15, model_name="kykim/bert-kor-base", shuffle=True, use_mlm=True):
    """
    원하는 문장 리스트를 주면 즉시 DataLoader를 반환합니다.
    """
    # 1. 내부에서 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. 데이터셋 생성 (문장 리스트 전달)
    dataset = Custom_Sentence_Dataset(
        sentences=sentences, 
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    # 3. 데이터 콜레이터 설정 (MLM 사용 여부에 따라 분기)
    if use_mlm:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=data_collator
        )
    else:
        # MLM 마스킹이 필요 없는 경우 (일반 평가나 embedding 추출 등)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
    
    return loader