import torch

rnn = torch.nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
out = torch.nn.Linear(20, 1)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

y = torch.tensor(
    [
        [
            [0],
        ],
        [
            [1],
        ]
    ], dtype=torch.float)

# (seq_len, batch, input_size)
h0 = torch.randn(2, 4, 20)
c0 = torch.randn(2, 4, 20)
#if batchfirst=Trues, (batch, seq_len, input_size)

input = torch.tensor(
    [
        [
            [1,1,1,1,1,1,1,1,1,1],
            [2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3],
            [4,4,4,4,4,4,4,4,4,4]
        ],
        [
            [5,5,5,5,5,5,5,5,5,5],
            [6,6,6,6,6,6,6,6,6,6],
            [7,7,7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7,7,7]
        ]
    ], dtype=torch.float)


for epoch in range(1000):
    output, (h_n, c_n) = rnn(input, None)

    # print(output)
    # print("############")
    # print(output[:, -1, :])

    output = out(output[:, -1, :]).view(2,1,1)
    # print(output)
    loss = loss_func(output, y)
    print("epoch: {}, loss: {}".format(epoch, loss))
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

input = torch.tensor(
    [
        [
            [1,1,1,1,1,1,1,1,1,1],
            [2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3],
            [4,4,4,4,4,4,4,4,4,4]
        ],
        # [
        #     [5,5,5,5,5,5,5,5,5,5],
        #     [6,6,6,6,6,6,6,6,6,6],
        #     [7,7,7,7,7,7,7,7,7,7],
        #     [7,7,7,7,7,7,7,7,7,7]
        # ],
    ]
, dtype=torch.float)


output, _ = rnn(input, None)

# print(output)
# print("###########")
# print(output[:, -1, :])

output = out(output[:, -1, :]).view(1,1,1)
print("res: ")
print(output)

y = torch.tensor(
    [
        [
            [0]
        ],
        # [
        #     [1]
        # ],
    ], dtype=torch.float)

loss = loss_func(output, y)
print("Loss: {}".format(loss))