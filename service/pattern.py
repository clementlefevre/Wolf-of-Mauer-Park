import pandas as pd

# Keltner Channel
def KELCH(df, n):

    df['KC_M_'+str(n)]=df[['High','Low','Close']].mean(axis=1).rolling(3).mean()
    df['KC_U_'+str(n)] = ((4 * df['High'] - 2 * df['Low'] + df['Close'])/3).rolling(3).mean()
    df['KC_D_'+str(n)] = ((-2 * df['High'] + 4 * df['Low'] + df['Close'])/3).rolling(3).mean()


    return df


# Donchian Channel
def DONCH(df, n):
    df['Donchian_High_'+str(n)] = df.High.rolling(n).max()
    df['Donchian_Low_'+str(n)] = df.Low.rolling(n).min()  
    return df


