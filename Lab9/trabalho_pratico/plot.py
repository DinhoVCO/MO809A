import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv', names=['server_round', 'cid', 'acc', 'loss'])
df_test  = pd.read_csv('evaluate.csv', names=['server_round', 'cid', 'acc', 'loss'])

fig, ax = plt.subplots(1, 2, figsize=(15, 5))


sns.lineplot(data=df_train, x='server_round', y='loss', ax=ax[0], color='b', label='Loss Treino')
sns.lineplot(data=df_train, x='server_round', y='acc', ax=ax[1], color='b', label='Acc Treino')
sns.lineplot(data=df_test, x='server_round', y='loss', ax=ax[0], color='r', label='Loss Teste')
sns.lineplot(data=df_test, x='server_round', y='acc', ax=ax[1], color='r', label='Acc Teste')



ax[0].set_title('Loss')
ax[1].set_title('Accuracy')

ax[0].grid(True, linestyle=':')
ax[1].grid(True, linestyle=':')

plt.savefig('output_plot.png')
