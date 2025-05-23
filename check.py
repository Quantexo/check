import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta
import pytz
import kaleido
import plotly.io as pio
import numpy as np

# --- Page Setup ---

st.set_page_config(page_title="Quantexo", layout="wide")

st.markdown(
    """ <style>
    .stApp {
    background-color: darkslategray;
    } </style>
    """,
    unsafe_allow_html=True
)
       
# --- SECTOR TO COMPANY MAPPING ---
sector_to_companies = {
    "Index": {"NEPSE"},
    "Sub-Index": {"BANKING", "DEVBANK", "FINANCE", "HOTELS", "HYDROPOWER", "INVESTMENT","LIFEINSU","MANUFACUTRE","MICROFINANCE","NONLIFEINSU", "OTHERS", "TRADING"},
    "Commercial Banks": {"ADBL","CZBIL","EBL","GBIME","HBL","KBL","LSL","MBL","NABIL","NBL","NICA","NIMB","NMB","PCBL","PRVU","SANIMA","SBI","SBL","SCB"},
    "Development Banks": {"CORBL","EDBL","GBBL","GRDBL","JBBL","KSBBL","LBBL","MDB","MLBL","MNBBL","NABBC","SADBL","SAPDBL","SHINE","SINDU"},
    "Finance": {"BFC","CFCL","GFCL","GMFIL","GUFL","ICFC","JFL","MFIL","MPFL","NFS","PFL","PROFL","RLFL","SFCL","SIFC"},
    "Hotels": {"CGH","CITY","KDL","OHL","SHL","TRH"},
    "Hydro Power": {"AHPC", "AHL", "AKJCL", "AKPL", "API", "BARUN", "BEDC", "BHDC", "BHPL", "BGWT", "BHL", "BNHC", "BPCL", "CHCL", "CHL", "CKHL", "DHPL", "DOLTI", "DORDI", "EHPL", "GHL", "GLH", "GVL", "HDHPC", "HHL", "HPPL", "HURJA", "IHL", "JOSHI", "KKHC", "KPCL", "KBSH", "LEC", "MAKAR", "MANDU", "MBJC", "MEHL", "MEL", "MEN", "MHCL", "MHNL", "MKHC", "MKHL", "MKJC", "MMKJL", "MHL", "MCHL", "MSHL", "NGPL", "NHDL", "NHPC", "NYADI", "PPL", "PHCL", "PMHPL", "PPCL", "RADHI", "RAWA", "RHGCL", "RFPL", "RIDI", "RHPL", "RURU", "SAHAS", "SHEL", "SGHC", "SHPC", "SIKLES", "SJCL", "SMH", "SMHL", "SMJC", "SPC", "SPDL", "SPHL", "SPL", "SSHL", "TAMOR", "TPC", "TSHL", "TVCL", "UHEWA", "ULHC", "UMHL", "UMRH", "UNHPL", "UPCL", "UPPER", "USHL", "USHEC", "VLUCL"},
    "Investment": {"CHDC","CIT","ENL","HATHY","HIDCL","NIFRA","NRN"},
    "Life Insurance":{"ALICL","CLI","CREST","GMLI","HLI","ILI","LICN","NLIC","NLICL","PMLI","RNLI","SJLIC","SNLI","SRLI"},
    "Manufacturing and Processing": {"BNL","BNT","GCIL","HDL","NLO","OMPL","SARBTM","SHIVM","SONA","UNL"},
    "Microfinance": {"ACLBSL","ALBSL","ANLB","AVYAN","CBBL","CYCL","DDBL","DLBS","FMDBL","FOWAD","GBLBS","GILB","GLBSL","GMFBS","HLBSL","ILBS","JBLB","JSLBB","KMCDB","LLBS","MATRI","MERO","MLBBL","MLBS","MLBSL","MSLB","NADEP","NESDO","NICLBSL","NMBMF","NMFBS","NMLBBL","NUBL","RSDC","SAMAJ","SHLB","SKBBL","SLBBL","SLBSL","SMATA","SMB","SMFBS","SMPDA","SWBBL","SWMF","ULBSL","UNLB","USLB","VLBS","WNLB"},
    "Non Life Insurance": {"HEI","IGI","NICL","NIL","NLG","NMIC","PRIN","RBCL","SALICO","SGIC"},
    "Others": {"HRL","MKCL","NRIC","NRM","NTC","NWCL"},
    "Trading": {"BBC","STC"}
}

#---UI LAYOUT---
col1, col2, col3, col4 = st.columns([0.5,0.5,0.5,0.5])

# --- Sector Selection ---
with col1:
    selected_sector = st.selectbox("Select Sector", options=[""]+ list(sector_to_companies.keys()), label_visibility="collapsed")

# ---Filter Companies based on Sector ---
with col2:
    if selected_sector:
        filtered_companies = sorted(sector_to_companies[selected_sector])
    else:
        filtered_companies = []
    
    selected_dropdown = st.selectbox(
        "Select Company",
        options=[""] + filtered_companies,
        label_visibility="collapsed",
        key="company"
    )

# ---Manual Input---
with col3:
    user_input = st.text_input(
        "🔍 Enter Company Symbol",
        "",
        label_visibility="collapsed",
        placeholder="🔍 Enter Symbol"
    )

with col4:
    col_search, col_scan = st.columns([1,1])
    with col_search:
        search_clicked = st.button("Search")

# --- Priority: Manual Entry Overrides Dropdown ---
if search_clicked:
    if user_input.strip():
        company_symbol = user_input.strip().upper()
        st.toast(f"🔎 Analyzing {company_symbol}...", icon="⚙️")
    elif selected_dropdown:
        company_symbol = selected_dropdown
        st.toast(f"🔎 Analyzing {company_symbol}...", icon="⚙️")
    else:
        st.warning("⚠️ Please enter or select a company.")
        st.stop()
else:
    company_symbol = ""

@st.cache_data(ttl=3600)
def get_sheet_data(symbol, sheet_name="Daily Price"):
    try:
        sheet_url = f"https://docs.google.com/spreadsheets/d/1Q_En7VGGfifDmn5xuiF-t_02doPpwl4PLzxb4TBCW0Q/export?format=csv&gid=0"  # Using gid=0 for the first sheet
        df = pd.read_csv(sheet_url)
        df = df.iloc[:, :7]
        df.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        
        # Get current time in Nepal timezone
        nepal_tz = pytz.timezone('Asia/Kathmandu')
        last_updated = datetime.now(nepal_tz)

        # Filter data based on company symbol
        df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
        filtered_df = df[df['symbol'].str.upper() == symbol.upper()]
        return filtered_df, last_updated
    except Exception as e:
        st.error(f"🔴 Error fetching data: {str(e)}")
        return pd.DataFrame(), None

def find_historical_resistance(df, current_price, swing_high):
    """Identify valid resistance levels from price history"""
    resistance_levels = []
    
    # Find previous swing highs that held as resistance
    peaks = df[(df['high'] > current_price) & (df['high'] < swing_high)]
    if not peaks.empty:
        # Cluster nearby peaks
        cluster_threshold = (swing_high - current_price) * 0.05
        clustered_peaks = []
        current_cluster = []
        
        for _, row in peaks.sort_values('high').iterrows():
            if not current_cluster or (row['high'] - current_cluster[-1] <= cluster_threshold):
                current_cluster.append(row['high'])
            else:
                clustered_peaks.append(np.mean(current_cluster))
                current_cluster = [row['high']]
        
        if current_cluster:
            clustered_peaks.append(np.mean(current_cluster))
        
        resistance_levels = sorted(list(set(clustered_peaks)))
    
    return resistance_levels

def detect_seller_absorption(df, min_targets=2, max_targets=12):
    signals = []
    df['absorption'] = False
    df['entry_price'] = None
    df['stop_loss'] = None
    df['targets'] = None
    
    # Calculate volume averages and ATR
    df['avg_volume'] = df['volume'].rolling(20).mean()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    for i in range(2, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Seller Absorption Criteria
        if (prev['open'] > prev['close'] and  # Bearish candle
            prev['volume'] > prev['avg_volume'] * 1.5 and  # High volume
            current['close'] > prev['open'] and  # Price moves above previous open
            current['volume'] > current['avg_volume'] * 1.2):  # Confirming volume
            
            # Only signal if no active absorption is pending
            if not df['absorption'].iloc[i-5:i].any():
                df.at[i, 'absorption'] = True
                
                # Calculate levels
                entry = current['close']
                swing_high = df['high'].iloc[max(0,i-20):i].max()
                swing_low = df['low'].iloc[max(0,i-20):i].min()
                atr = df['atr'].iloc[i]

                # Calculate targets (based on historical resistance zones)
                targets = []
                # Find meaningful historical resistance levels
                resistance_levels = find_historical_resistance(df[:i], entry, swing_high)
                
                # Use found resistance levels or create Fibonacci-based ones
                if len(resistance_levels) >= min_targets:
                    targets = sorted(resistance_levels[:max_targets])
                else:
                    # Fallback to Fibonacci levels if not enough resistance zones found
                    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618, 2.0, 2.618, 3.0]
                    price_range = swing_high - entry
                    targets = [entry + (price_range * level) for level in fib_levels[:max_targets]]
                
                # Validate targets
                targets = [t for t in targets if t > entry][:max_targets]
                if not targets:
                    continue

                # NOW calculate other values that depend on targets
                swing_low = df['low'].iloc[max(0,i-20):i].min()
                atr = df['atr'].iloc[i]
                conservative_entries = [
                    entry + (swing_high - entry) * 0.236,
                    entry + (swing_high - entry) * 0.382,
                    entry - (entry - swing_low) * 0.618
                ]
                hit_dates = [None] * len(targets)  # Now targets is defined

                # Determine stop loss (below recent swing low)
                stop_loss = swing_low - (atr * 0.5)

                signals.append({
                    'date': current['date'],
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'targets': targets,
                    'conservative_entries': conservative_entries,
                    'hit_dates': hit_dates,
                    'hit_stop': False,
                    'hit_targets': [False] * len(targets)
                })
                
                # Update dataframe
                df.at[i, 'entry_price'] = entry
                df.at[i, 'stop_loss'] = stop_loss
                df.at[i, 'targets'] = targets
    # NEW CODE: Analyze which targets were hit
    for signal in signals:
        subsequent_data = df[df['date'] > signal['date']]
        
        # Check if stop loss was hit
        stop_hits = subsequent_data[subsequent_data['low'] <= signal['stop_loss']]
        if not stop_hits.empty:
            signal['hit_stop'] = True
            signal['stop_hit_date'] = stop_hits.iloc[0]['date']
        
        # Check which targets were hit
        for i, target in enumerate(signal['targets']):
            target_hits = subsequent_data[subsequent_data['high'] >= target]
            if not target_hits.empty:
                signal['hit_targets'][i] = True
                signal['hit_dates'][i] = target_hits.iloc[0]['date']
    return df, signals

# Add this function to format percentage changes
def format_pct_change(entry, price):
    pct = ((price - entry) / entry) * 100
    return f"({abs(pct):.2f}%)"

def plot_absorption_signals(fig, df, signals):
    """Add absorption signals to the chart with formatted summary table - ONLY MOST RECENT TRADE"""
    table_content = ["<b>SELLER ABSORPTION TRADE</b>"]
    latest_date = df['date'].max()

    # Filter to get only active signals (not hit stop loss)
    active_signals = [signal for signal in signals if not signal['hit_stop']]
    
    # If we have active signals, get the most recent one
    if active_signals:
        # Sort by date and get the most recent signal
        most_recent_signal = max(active_signals, key=lambda x: x['date'])
        
        # Entry section
        table_content.extend([
            f"<b>Aggressive Entry</b> = {most_recent_signal['entry']:.2f} ({most_recent_signal['date'].strftime('%b %d, %Y')})",
            f"<b>Conservative Entry</b> = {most_recent_signal['conservative_entries'][0]:.2f}, {most_recent_signal['conservative_entries'][1]:.2f}, {most_recent_signal['conservative_entries'][2]:.2f}"
        ])
        
        # Targets section
        targets_text = []
        for i, (target, hit_date) in enumerate(zip(most_recent_signal['targets'], most_recent_signal['hit_dates'])):
            status = f"HIT on {hit_date.strftime('%b %d, %Y')}" if hit_date else ""
            pct = format_pct_change(most_recent_signal['entry'], target)
            targets_text.append(f"- TP {i+1} = {target:.2f} {pct} {status}")
        
        # Stop loss section
        sl_pct = format_pct_change(most_recent_signal['entry'], most_recent_signal['stop_loss'])
        table_content.extend(targets_text + ["", f"<b>Stop Loss</b> = {most_recent_signal['stop_loss']:.2f} {sl_pct}"])
    else:
        table_content.append("No active seller absorption trades found.")

    # Add summary table
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.03, y=0.97,
        text="<br>".join(table_content),
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0)",  # Changed to visible background
        font=dict(color="white", size=12, family="Courier New, monospace")
    )
    return fig

def detect_signals(df):
    results = []
    df['point_change'] = df['close'].diff().fillna(0)
    df['tag'] = ''

    min_window = min(20, max(5, len(df) // 2)) 
    avg_volume = df['volume'].rolling(window=min_window).mean().fillna(method='bfill').fillna(df['volume'].mean())

    for i in range(min(3, len(df)-1), len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        next_candles = df.iloc[i + 1:min(i + 6, len(df))]
        body = abs(row['close'] - row['open'])
        prev_body = abs(prev['close'] - prev['open'])
        recent_tags = df['tag'].iloc[max(0, i - 9):i]
        
        if (
            row['close'] > row['open'] and
            row['volume'] > avg_volume[i] * 1.2
        ):
            df.loc[df['tag'] == '⛔', 'tag'] = ''
            for j, candle in next_candles.iterrows():
                if candle['close'] < row['open']:
                    df.at[j, 'tag'] = '⛔'
                    break
        if (
            row['open'] > row['close'] and
            row['volume'] > avg_volume[i] * 1.2
        ):
            df.loc[df['tag'] == '🚀', 'tag'] = ''
            for j, candle in next_candles.iterrows():
                if candle['close'] > row['open']:
                    df.at[j, 'tag'] = '🚀'
                    break
        if (
            row['close'] > row['open'] and
            row['close'] >= row['high'] - (row['high'] - row['low']) * 0.1 and
            row['volume'] > avg_volume[i] * 2 and
            body > prev_body and
            '🟢' not in recent_tags.values
        ):
            df.at[i, 'tag'] = '🟢'
        if (
            row['open'] > row['close'] and
            row['close'] <= row['low'] + (row['high'] - row['low']) * 0.1 and
            row['volume'] > avg_volume[i] * 2 and
            body > prev_body and
            '🔴' not in recent_tags.values
        ):
            df.at[i, 'tag'] = '🔴'
        if (
            i >= 10 and
            row['close'] > max(df['high'].iloc[i - 10:i]) and
            row['volume'] > avg_volume[i] * 1.8
        ):
            if not (df['tag'].iloc[i - 8:i] == '💥').any():
                df.at[i, 'tag'] = '💥'
        if (
            i >= 10 and
            row['close'] < min(df['low'].iloc[i - 10:i]) and
            row['volume'] > avg_volume[i] * 1.8
        ):
            if not (df['tag'].iloc[i - 8:i] == '💣').any():
                df.at[i, 'tag'] = '💣'
        if (
            row['close'] > row['open'] and
            body > (row['high'] - row['low']) * 0.85 and
            row['volume'] > avg_volume[i] * 2
        ):
            df.at[i, 'tag'] = '🐂'
        if (
            row['open'] > row['close'] and
            body > (row['high'] - row['low']) * 0.85 and
            row['volume'] > avg_volume[i] * 2
        ):
            df.at[i, 'tag'] = '🐻'

        if df.at[i, 'tag']:
            results.append({
                'symbol': row['symbol'],
                'tag': df.at[i, 'tag'],
                'date': row['date'].strftime('%Y-%m-%d')
            })
    return results

if company_symbol:
    sheet_name = "Daily Price"
    df, last_updated = get_sheet_data(company_symbol, sheet_name)

    if df.empty:
        st.warning(f"No data found for {company_symbol}")
        st.stop()

    try:
        # Convert column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Check required columns
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            st.error("❌ Missing required columns: date, open, high, low, close, volume")
            st.stop()

        # Convert and validate dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isnull().any():
            st.error("❌ Invalid date format in some rows")
            st.stop()

        # Validate numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('[^\d.]', '', regex=True),  # Remove non-numeric chars
                errors='coerce'
            )
            if df[col].isnull().any():
                bad_rows = df[df[col].isnull()][['date', col]].head()
                st.error(f"❌ Found {df[col].isnull().sum()} invalid values in {col} column. Examples:")
                st.dataframe(bad_rows)
                st.stop()

        # Remove any rows with NA values
        df = df.dropna()
        if len(df) == 0:
            st.error("❌ No valid data after cleaning")
            st.stop()

        # Sort and reset index
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Detect signals
        results = detect_signals(df)
        # Detect seller absorption patterns
        
        fig = go.Figure()
        # Add watermark annotation first (behind everything)
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text=f"Quantexo<br>{company_symbol}",
            showarrow=False,
            font=dict(
                size=40,
                color="rgba(128,128,128,0.2)"  # Semi-transparent gray
            ),
            align="center",
        )
        df, all_absorptions = detect_seller_absorption(df)

        if all_absorptions:
            fig = plot_absorption_signals(fig, df, all_absorptions)
            
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['close'],
            mode='lines', name='Close Price',
            line=dict(color='lightblue', width=2),
            customdata=df[['date', 'open', 'high', 'low', 'close', 'point_change']],
            hovertemplate=(
                "📅 Date: %{customdata[0]|%Y-%m-%d}<br>" +
                "🟢 Open: %{customdata[1]:.2f}<br>" +
                "📈 High: %{customdata[2]:.2f}<br>" +
                "📉 Low: %{customdata[3]:.2f}<br>" +
                "🔚 LTP: %{customdata[4]:.2f}<br>" +
                "📊 Point Change: %{customdata[5]:.2f}<extra></extra>"
            )
        ))  

        tag_labels = {
            '🟢': '🟢 Aggressive Buyers',
            '🔴': '🔴 Aggressive Sellers',
            '⛔': '⛔ Buyer Absorption',
            '🚀': '🚀 Seller Absorption',
            '💥': '💥 Bullish POR',
            '💣': '💣 Bearish POR',
            '🐂': '🐂 Bullish POI',
            '🐻': '🐻 Bearish POI'
        }

        signals = df[df['tag'] != '']
        for tag in signals['tag'].unique():
            subset = signals[signals['tag'] == tag]
            fig.add_trace(go.Scatter(
                x=subset['date'], y=subset['close'],
                mode='markers+text',
                name=tag_labels.get(tag, tag),
                text=[tag] * len(subset),
                textposition='top center',
                textfont=dict(size=20),
                marker=dict(size=14, symbol="circle", color='white'),
                customdata=subset[['open', 'high', 'low', 'close', 'point_change']].values,
                hovertemplate=(
                    "📅 Date: %{x|%Y-%m-%d}<br>" +
                    "🟢 Open: %{customdata[0]:.2f}<br>" +
                    "📈 High: %{customdata[1]:.2f}<br>" +
                    "📉 Low: %{customdata[2]:.2f}<br>" +
                    "🔚 LTP: %{customdata[3]:.2f}<br>" +
                    "📊 Point Change: %{customdata[4]:.2f}<br>" +
                    f"{tag_labels.get(tag, tag)}<extra></extra>"
                )
            ))
        
        # Calculate 20 days ahead of the last date
        last_date = df['date'].max()
        extended_date = last_date + timedelta(days=20)
        fig.update_layout(
            height=800,
            width=1800,
            plot_bgcolor="darkslategray",
            paper_bgcolor="darkslategray",
            font_color="white",
            xaxis=dict(title="Date", tickangle=-45, showgrid=False, range=[(df['date'].max() - pd.Timedelta(days=365)), extended_date]), #extend x-axis to show space after latest date
            yaxis=dict(title="Price", showgrid=False, zeroline=True, zerolinecolor="gray", autorange=True),
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,  # Adjust this value to move further down if needed
                xanchor="center",
                x=0.5,
                font=dict(size=14),
                bgcolor="rgba(0,0,0,0)"  # Optional: keeps legend background transparent)
            ),
            # Add zoom and pan capabilities
            dragmode="zoom",  # Enable box zoom
        )
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        df['date'] = pd.to_datetime(df['date'])
        last_data_date = df['date'].max().strftime("%Y-%m-%d")
        if last_updated:
            formatted_time = last_updated.strftime("%Y-%m-%d %H:%M:%S")
            cols = st.columns(2)
            cols[0].caption(f"⏱️ Data fetched: {formatted_time}")
            cols[1].caption(f"📅 Latest data point: {last_data_date}")
        st.plotly_chart(fig, use_container_width=False)
    except Exception as e:
        st.error(f"⚠️ Processing error: {str(e)}")
else:
    st.info("ℹ👆🏻 Enter a company symbol to get analysed chart 👆🏻")