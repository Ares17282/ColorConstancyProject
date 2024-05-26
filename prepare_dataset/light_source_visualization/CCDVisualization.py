import numpy as np
import plotly.graph_objs as go

# 加载数据
data = np.load('/home/zhouxinlei/ColorConstancy/CCD/labels568.npy')
print('散点数:',len(data))

# 创建散点图
fig = go.Figure(data=[go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], mode='markers', marker=dict(color='blue', size=2))])

# 设置图形布局
fig.update_layout(scene=dict(xaxis=dict(nticks=10, range=[0, 1], title='R'),
                             yaxis=dict(nticks=10, range=[0, 1], title='G'),
                             zaxis=dict(nticks=10, range=[0, 1], title='B'),
                             aspectmode='cube'))

# 显示图形
fig.show()
