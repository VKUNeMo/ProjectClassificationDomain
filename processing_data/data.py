from getLexicalFeature import *
import pandas as pd
import time

start_time = time.time()


path = 'main/processing_data/dataset_70K.csv'
df = pd.read_csv(path)
urls = df.iloc[:, 0].tolist()
labels = df.iloc[:,1].tolist()
length, entropy, percent, num_special, bao_chi, chinh_tri, co_bac, khieu_dam, tpmt, con_lai = [
    [] for _ in range(10)]
for url in urls:
    features = getLexicalInputNN(url)
    [l.append(f) for l, f in zip([length, entropy, percent, num_special, bao_chi,
                                  chinh_tri, co_bac, khieu_dam, tpmt, con_lai], features)]
df = pd.DataFrame({
    'url': urls,
    'length': length,
    'entropy': entropy,
    'percent_num': percent,
    'num_special': num_special, 
    'bao_chi': bao_chi,
    'chinh_tri': chinh_tri,
    'co_bac': co_bac,
    'khieu_dam': khieu_dam,
    'tpmt': tpmt,
    'con_lai': con_lai,
    'label': labels
})


xlsx_file = 'data_train_leven.csv'
df.to_csv(xlsx_file, index=False)
print(f'Data has been written to {xlsx_file}')


end_time = time.time()
# Calculate execution time
execution_time = end_time - start_time
print("Execution time:", execution_time/60, "minutes")
