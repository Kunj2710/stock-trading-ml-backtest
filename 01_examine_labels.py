import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('KO_weekly_return_volatility.csv')

df['mu'] = df['mean_return'].round(2)
df['sigma'] = df['volatility'].round(2)

df['color'] = df['decision'].map({'YES': 'green', 'NO': 'red'})

years = [2021, 2022, 2023, 2024, 2025]

for year in years:

    fig, ax = plt.subplots(figsize=(10, 7))

    yr_data = df[df['Year'] == year].copy()

    green = yr_data[yr_data['decision'] == 'YES']
    red   = yr_data[yr_data['decision'] == 'NO']

    ax.scatter(green['mu'], green['sigma'],
               c='green', s=200,
               edgecolors='black', linewidths=0.5,
               label='YES (Green)', zorder=3)

    ax.scatter(red['mu'], red['sigma'],
               c='red', s=200,
               edgecolors='black', linewidths=0.5,
               label='NO (Red)', zorder=3)

    for _, row in yr_data.iterrows():
        ax.annotate(str(int(row['Week_Number'])),
                    (row['mu'], row['sigma']),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords='offset points')

    ax.set_xlabel('µ - Mean Return (%)', fontsize=12)
    ax.set_ylabel('σ - Volatility (%)',  fontsize=12)
    ax.set_title(f'KO Stock Labels - Year {year}', fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')  # horizontal line at 0
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')  # vertical line at 0
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f'KO_year_{year}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved chart for {year}")