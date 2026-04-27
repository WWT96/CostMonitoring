import pandas as pd
import random
from datetime import datetime, timedelta

# 生成测试数据
items = ['A001', 'B002', 'C003', 'D004']
data = []

start_date = datetime(2023, 1, 1)

for item in items:
    # 每个物料生成 10 条记录
    current_price = random.uniform(10, 100)
    for i in range(10):
        date = start_date + timedelta(days=i*30 + random.randint(0, 10))
        date_str = date.strftime('%Y-%m-%d')
        
        # 价格波动
        current_price = current_price * (1 + random.uniform(-0.1, 0.1))
        
        # 模拟“成本有效期”列，可能包含额外文字
        validity_str = f"有效期至 {date_str} (备注)"
        
        data.append({
            "物料编码": item,
            "成本有效期": validity_str,
            "价格": round(current_price, 2)
        })

df = pd.DataFrame(data)

# 保存为 Excel
df.to_excel("test_data.xlsx", index=False)
print("测试数据已生成: test_data.xlsx")

# 保存为 CSV
df.to_csv("test_data.csv", index=False, encoding='utf-8-sig')
print("测试数据已生成: test_data.csv")
