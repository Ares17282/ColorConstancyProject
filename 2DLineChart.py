import numpy as np
import plotly.graph_objects as go

data = np.load('resultModel/tendency_train.npy')
data1 = np.load('resultModel/tendency_test.npy')

data = data.astype(float)
data1 = data1.astype(float)

x_data = data[:, 0]
y_data = data[:, 1]
y1_data = data1[:, 1]

figure1 = go.Scatter(x=x_data, y=y_data, mode='lines')
figure2 = go.Scatter(x=x_data, y=y1_data, mode='lines')

# 创建折线图
fig = go.Figure(data=[figure1, figure2])

# 添加标题和标签
fig.update_layout(
    title='Training - Validation',
    xaxis_title='Step',
    yaxis_title='Loss'
)

# 显示图形
fig.show()
