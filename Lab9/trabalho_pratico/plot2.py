import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv', names=['server_round', 'cid', 'acc', 'loss'])
df_test  = pd.read_csv('evaluate.csv', names=['server_round', 'cid', 'acc', 'loss'])

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
sns.histplot(x=df_test['acc'].values[-10:], kde=True, color='r', bins=10, ax=ax[0])
sns.barplot(x=df_test['cid'].values[-10:], y=df_test['acc'].values[-10:], color='b', ec='k', ax=ax[1])

ax[0].set_title('Distribuição de Acurácia dos Clientes')
ax[0].set_ylabel('Quantidade de Clientes')
ax[0].set_xlabel('Acurácia Teste(%)')

ax[1].set_title('Acurácia por Cliente')
ax[1].set_ylabel('Acurácia Teste')
ax[1].set_xlabel('Client ID (#)')

for _ in range(2):
  ax[_].grid(True, linestyle=':')
  ax[_].set_axisbelow(True)

plt.savefig('output_plot2NIID.png')
