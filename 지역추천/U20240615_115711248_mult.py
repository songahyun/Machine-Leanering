import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f'using {device}')

# 데이터 로드 및 전처리
def load_data():
    preprocessed_data = pd.read_csv('processed_data.csv')
    preprocessed_data = preprocessed_data.astype(np.float32)

    # INTRST_VALUE 컬럼 확인
    y_columns = [col for col in preprocessed_data.columns if 'INTRST_VALUE' in col]
    preprocessed_data[y_columns]

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, preprocessed_data[y_columns], test_size=0.2, random_state=42)

    # Tensor로 변환
    X_train_tensor = torch.tensor(X_train.values)
    y_train_tensor = torch.tensor(y_train.values)
    X_test_tensor = torch.tensor(X_test.values)
    y_test_tensor = torch.tensor(y_test.values)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, y_columns, preprocessed_data

# DataLoader 생성
def create_dataloader(X_train_tensor, y_train_tensor, batch_size=64):
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# 오토인코더 모델 정의
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, 11)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 훈련 함수 정의
def train_autoencoder():
    # Initialize W&B
    wandb.init()
    config = wandb.config
    # 데이터 로드
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, y_columns, preprocessed_data = load_data()

    # DataLoader 생성
    train_loader = create_dataloader(X_train_tensor, y_train_tensor)

    # Interest value 컬럼 인덱스 추출
    interest_value_columns = [col for col in preprocessed_data.columns if 'INTRST_VALUE' in col]
    interest_value_indices = [preprocessed_data.columns.get_loc(col) for col in interest_value_columns]
    I = interest_value_indices[0]
    J = interest_value_indices[-1] + 1
    # 모델 초기화
    input_dim = X_train_tensor.shape[1]
    model = Autoencoder(input_dim, config.hidden_dim1, config.hidden_dim2, config.latent_dim)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    num_epochs = config.num_epochs
    mask_prob = config.start_mask_prob
    mask_prob_increment = (config.end_mask_prob - config.start_mask_prob) / (num_epochs / (config.mask_prob_step + 1) * config.finish_update)
    for epoch in tqdm(range(num_epochs)):
        if epoch > 0 and epoch % config.mask_prob_step == 0 and epoch < num_epochs / 2 + 1:
            mask_prob += mask_prob_increment

        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            mask = torch.rand(inputs[:, I:J].shape) < mask_prob
            inputs[:, I:J][mask] = 0

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log the loss to W&B
        wandb.log({"loss": loss.item(), "epoch": epoch})
    # Test data masking
    test_mask_prob = 0.7
    X_test_masked = X_test_tensor.clone().to(device)
    X_test_origin = X_test_tensor.clone().to(device)

    mask = torch.rand(X_test_masked[:, I:J].shape) < test_mask_prob
    X_test_masked[:, I:J][mask] = 0
    # 마스킹되지 않은 값 유지
    X_test_origin[:, I:J] = X_test_tensor[:, I:J] * ~mask
    # 복원된 데이터 예측
    model.eval()
    with torch.no_grad():
        restored_data = model(X_test_masked).cpu().numpy()

    restored_fill_data = np.copy(restored_data)
    
    # 복원된 데이터 확인
    restored_df = pd.DataFrame(restored_data, columns=interest_value_columns)
    for i in range(X_test_tensor.shape[0]):
        for j in range(I, J):
            if not mask[i, j - I]:  # 마스킹되지 않은 값일 때
                restored_fill_data[i, j - I] = y_test_tensor[i, j - I]

    # 정확도 평가
    restored_mse = mean_squared_error(y_test_tensor, restored_df)
    restored_mae = mean_absolute_error(y_test_tensor, restored_df)


    wandb.log({"mean_squared_error": restored_mse, "mean_absolute_error": restored_mae})

    wandb.run.summary["mean_squared_error"] = restored_mse
    wandb.run.summary["mean_absolute_error"] = restored_mae

    return model, config

if __name__ == "__main__":
    train_autoencoder()