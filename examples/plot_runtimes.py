
import matplotlib.pyplot as plt
import pandas as pd
import io

data = """
Method              | Energy (Ha)  | Δ vs FCI (mHa) | Runtime (s)
--------------------+--------------+----------------+------------
RHF                 | -74.96294666 | +49.491        | 0.553
MP2                 | -74.99844949 | +13.988        | 4.478
CCSD                | -75.01232107 | +0.116         | 0.856
CASCI (full)        | -75.01243743 | +0.000         | 1.120
FCI (full)          | -75.01243743 | +0.000         | 0.0
SQD (full) [ucj]    | -75.01235001 | +0.087         | 2.712
SQD (active) [ucj]  | -75.01229993 | +0.138         | 1.811
SQD (full) [lucj]   | -75.01239347 | +0.044         | 2.879
SQD (active) [lucj] | -75.01229944 | +0.138         | 1.893
SQD (full) [he]     | -75.01243743 | +0.000         | 2.970
SQD (active) [he]   | -75.01235929 | +0.078         | 1.981
SQD (full) [hf]     | -74.96294666 | +49.491        | 3.763
SQD (active) [hf]   | -74.96294666 | +49.491        | 4.465
CASCI (active)      | -75.01235929 | +0.078         | 0.587
"""

df = pd.read_csv(io.StringIO(data), sep='|', skipinitialspace=True, header=0, skip_blank_lines=True)

df.columns = [col.strip() for col in df.columns]

df = df.iloc[1:].copy()

df['Method'] = df['Method'].str.strip()
df['Runtime (s)'] = pd.to_numeric(df['Runtime (s)'].astype(str).str.strip(), errors='coerce')

df_sqd = df[df['Method'].str.contains('SQD')].copy()

if not df_sqd.empty:
    
    df_sqd['Ansatz'] = df_sqd['Method'].str.extract(r'\[(.*?)\]')
    df_sqd['Type'] = df_sqd['Method'].str.extract(r'\((.*?)\)')[0].str.split(',').str[0]

    
    pivot_df = df_sqd.pivot(index='Ansatz', columns='Type', values='Runtime (s)')

    
    pivot_df.plot(kind='bar', figsize=(12, 7))

    plt.title('SQD Ansatz Runtime Comparison')
    plt.ylabel('Runtime (s)')
    plt.xlabel('Ansatz')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('examples/runtimes_comparison.png')
    plt.close()

    print("Graph saved to examples/runtimes_comparison.png")
else:
    print("No SQD data found to plot.")

