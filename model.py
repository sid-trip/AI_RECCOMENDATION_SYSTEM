import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
EMBEDDING_DIM = 32

df = pd.read_csv('ML_DATA/ml-latest-small/ratings.csv')

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

df['user_idx'] = user_encoder.fit_transform(df['userId'])
df['movie_idx'] = movie_encoder.fit_transform(df['movieId'])

num_users = df['user_idx'].nunique()
num_movies = df['movie_idx'].nunique()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

class MovieLensDataset(Dataset):
    def __init__(self, user_ids, movie_ids, ratings):
        self.users = torch.tensor(user_ids, dtype=torch.long)
        self.movies = torch.tensor(movie_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

train_dataset = MovieLensDataset(
    train_df['user_idx'].values, 
    train_df['movie_idx'].values, 
    train_df['rating'].values
)
test_dataset = MovieLensDataset(
    test_df['user_idx'].values, 
    test_df['movie_idx'].values, 
    test_df['rating'].values
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, user_idx, movie_idx):
        user_embed = self.user_embedding(user_idx)
        movie_embed = self.movie_embedding(movie_idx)
        
        x = torch.cat([user_embed, movie_embed], dim=1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.output(x)
        
        return x

model = NeuralCollaborativeFiltering(num_users, num_movies, EMBEDDING_DIM).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for user_ids, movie_ids, ratings in train_loader:
        user_ids = user_ids.to(device)
        movie_ids = movie_ids.to(device)
        ratings = ratings.to(device)
        
        optimizer.zero_grad()
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions.squeeze(), ratings)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {avg_loss:.4f}")

model.eval()
test_loss = 0

with torch.no_grad():
    for user_ids, movie_ids, ratings in test_loader:
        user_ids, movie_ids = user_ids.to(device), movie_ids.to(device)
        ratings = ratings.to(device)
        
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions.squeeze(), ratings)
        test_loss += loss.item()

print(f"Final Test MSE Loss: {test_loss / len(test_loader):.4f}")

torch.save(model.state_dict(), "ncf_model.pth")