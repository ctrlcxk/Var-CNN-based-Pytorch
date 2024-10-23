import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DeepFingerprintingModel(nn.Module):
    def __init__(self, config):
        super(DeepFingerprintingModel, self).__init__()
        
        num_mon_sites = config['num_mon_sites']
        num_unmon_sites_test = config['num_unmon_sites_test']
        num_unmon_sites_train = config['num_unmon_sites_train']
        num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train
        seq_length = config['seq_length']
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, 8, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=4)
        self.drop1 = nn.Dropout(0.1)
        
        # Block 2
        self.conv3 = nn.Conv1d(32, 64, 8, stride=1, padding=4)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, 8, stride=1, padding=4)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=4)
        self.drop2 = nn.Dropout(0.1)
        
        # Block 3
        self.conv5 = nn.Conv1d(64, 128, 8, stride=1, padding=4)
        self.bn5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 128, 8, stride=1, padding=4)
        self.bn6 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=8, stride=4)
        self.drop3 = nn.Dropout(0.1)
        
        # Block 4
        self.conv7 = nn.Conv1d(128, 256, 8, stride=1, padding=4)
        self.bn7 = nn.BatchNorm1d(256)
        self.conv8 = nn.Conv1d(256, 256, 8, stride=1, padding=4)
        self.bn8 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=8, stride=4)
        self.drop4 = nn.Dropout(0.1)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * (seq_length // (4 ** 4)), 512)  # Adjust size according to input sequence length
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.drop_fc2 = nn.Dropout(0.5)

        output_classes = num_mon_sites if num_unmon_sites == 0 else num_mon_sites + 1
        self.fc3 = nn.Linear(512, output_classes)

    def forward(self, x):
        # Block 1
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.drop4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop_fc2(x)
        
        # Output layer
        x = F.softmax(self.fc3(x), dim=1)
        return x

# Configuration example
config = {
    'num_mon_sites': 100,
    'num_mon_inst_test': 50,
    'num_mon_inst_train': 100,
    'num_unmon_sites_test': 50,
    'num_unmon_sites_train': 100,
    'seq_length': 5000
}

# Instantiate model
model = DeepFingerprintingModel(config)

# Define optimizer and loss function
optimizer = optim.Adamax(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

# Example input: batch of size 16, sequence length from config['seq_length'], 1 channel
input_data = torch.randn(16, 1, config['seq_length'])
output = model(input_data)

# Print model output
print(output)
