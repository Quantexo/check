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

def find_historical_resistance(df, current_price, swing_high, lookback_periods = 100):
    """Identify valid resistance levels from price history"""
    resistance_levels = []
    
    # Look at more historical data for better resistance detection
    historical_data = df.tail(lookback_periods) if len(df) > lookback_periods else df

    # Method 1: Find previous swing highs (more flexible criteria)
    # Look for highs that are above current price (potential resistance)
    potential_resistance = historical_data[historical_data['high'] > current_price]

    if not potential_resistance.empty:
        # Get unique high values and sort them
        high_values = sorted(potential_resistance['high'].unique())
        
        # Cluster nearby resistance levels
        cluster_threshold = current_price * 0.02  # 2% clustering threshold
        clustered_levels = []
        current_cluster = []

        for high_val in high_values:
            if not current_cluster or (high_val - current_cluster[-1] <= cluster_threshold):
                current_cluster.append(high_val)
            else:
                # Add average of current cluster
                clustered_levels.append(np.mean(current_cluster))
                current_cluster = [high_val]
        
        # Don't forget the last cluster
        if current_cluster:
            clustered_levels.append(np.mean(current_cluster))
        
        resistance_levels.extend(clustered_levels)
    
    # Method 2: Add psychological levels (round numbers)
    # Find round numbers above current price
    price_magnitude = 10 ** (len(str(int(current_price))) - 1)  # e.g., 100 for prices like 150-999
    
    psychological_levels = []
    for multiplier in [1, 2, 5, 10, 15, 20, 25, 50]:
        level = price_magnitude * multiplier
        if level > current_price and level <= swing_high * 2:  # Don't go too far above swing high
            psychological_levels.append(level)
    
    resistance_levels.extend(psychological_levels)
    
    # Remove duplicates and sort
    resistance_levels = sorted(list(set(resistance_levels)))
    
    return resistance_levels

def detect_seller_absorption(df, min_targets=3, max_targets=15):
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
            prev['volume'] > prev['avg_volume'] * 2 and  # High volume
            current['close'] > prev['open'] and  # Price moves above previous open
            current['volume'] > current['avg_volume'] * 2):  # Confirming volume
            
            print(f"🔥 Signal detected on: {current['date']} (Row {i})")  # Debug print

            # Calculate price gain from recent low
            recent_low = df['low'].iloc[max(0, i - 60):i].min()
            price_gain_pct = (current['close'] - recent_low) / recent_low

            # Skip signal if price has already gained more than 60%
            if price_gain_pct > 0.60:
                continue

            # Only signal if no active absorption is pending (check last 5 bars)
            if not df['absorption'].iloc[max(0, i-5):i].any():
                # Clear any previous 🚀 tags
                df.loc[df['tag'] == '🚀', 'tag'] = ''
                df.at[i, 'absorption'] = True
                df.at[i, 'tag'] = '🚀'
                
                # Calculate levels
                entry = current['close']
                swing_high = df['high'].iloc[max(0,i-20):i].max()
                swing_low = df['low'].iloc[max(0,i-20):i].min()
                atr = df['atr'].iloc[i]

                # Calculate stop loss (max 8% from entry)
                max_sl_pct = entry * 0.92  # 8% below entry
                proposed_sl = min(swing_low, max_sl_pct)  # Take the tighter stop
                
                # Ensure stop isn't too tight (minimum 3% below entry)
                min_sl_pct = entry * 0.97
                stop_loss = max(proposed_sl, min_sl_pct)
                
                # Calculate targets (focused on 15-20% gains)
                targets = []
                
                # Method 1: Find nearby historical resistance (within 20%)
                resistance_levels = find_historical_resistance(df[:i], entry, swing_high)
                nearby_resistance = [r for r in resistance_levels if (r > entry) and (r <= entry * 1.20)]
                
                if nearby_resistance:
                    targets.extend(nearby_resistance[:max_targets])
                
                # Method 2: Use tighter Fibonacci levels if needed
                if len(targets) < max_targets:
                    fib_levels = [0.05, 0.10, 0.15, 0.20]  # Only near-term levels
                    price_range = max(atr * 3, entry * 0.10)  # Conservative range
                    fib_targets = [entry + (price_range * level) for level in fib_levels]
                    targets.extend(fib_targets)
                
                # Method 3: Simple percentage targets as fallback
                if len(targets) < max_targets:
                    pct_targets = [
                        entry * 1.05,
                        entry * 1.10,
                        entry * 1.15,
                        entry * 1.20
                    ]
                    targets.extend(pct_targets)
                
                # Finalize targets (remove duplicates, sort, and limit)
                targets = sorted(list(set([t for t in targets if t > entry])))
                targets = targets[:max_targets]
                
                # Conservative entries (closer to stop loss)
                conservative_entries = [
                    entry * 0.99,  # 1% below
                    entry * 0.98,  # 2% below
                    stop_loss * 1.01  # Just above stop
                ]
                
                hit_dates = [None] * len(targets)
                
                signals.append({
                    'date': current['date'],
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'targets': targets,
                    'conservative_entries': conservative_entries,
                    'hit_dates': hit_dates,
                    'hit_stop': False,
                    'hit_targets': [False] * len(targets),
                    'is_current_signal': True  # Mark as the current signal
                })
                
                # Mark all previous signals as not current
                for prev_signal in signals[:-1]:
                    prev_signal['is_current_signal'] = False
                
                # Update dataframe
                df.at[i, 'entry_price'] = entry
                df.at[i, 'stop_loss'] = stop_loss
                df.at[i, 'targets'] = targets
    
    # Analyze which targets were hit
    for signal in signals:
        subsequent_data = df[df['date'] > signal['date']]
        
        # Check if stop loss was hit
        stop_hits = subsequent_data[subsequent_data['low'] <= signal['stop_loss']]
        if not stop_hits.empty:
            signal['hit_stop'] = True
            signal['stop_hit_date'] = stop_hits.iloc[0]['date']
        
        # Check which targets were hit (within 20 trading days)
        subsequent_data = subsequent_data.head(20)
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
    """Add absorption signals to the chart with formatted summary table - SHOWS CURRENT TAGGED TRADE"""
    table_content = ["<b>SELLER ABSORPTION TRADE</b>"]
    
    # Find the signal that corresponds to the currently tagged 🚀 on the chart
    current_signal = None
    
    # First, try to find the signal marked as current
    for signal in signals:
        if signal.get('is_current_signal', False):
            current_signal = signal
            break
    
    # If no current signal found, get the most recent signal (whether active or not)
    if current_signal is None and signals:
        current_signal = max(signals, key=lambda x: x['date'])
    
    if current_signal:
        # Determine trade status
        trade_status = ""
        if current_signal['hit_stop']:
            trade_status = f" [STOPPED OUT on {current_signal['stop_hit_date'].strftime('%b %d, %Y')}]"
        elif any(current_signal['hit_targets']):
            hit_count = sum(current_signal['hit_targets'])
            trade_status = f" [ACTIVE - {hit_count} targets hit]"
        else:
            trade_status = " [ACTIVE]"
        
        # Entry section
        table_content.extend([
            f"<b>Aggressive Entry</b> = {current_signal['entry']:.2f} ({current_signal['date'].strftime('%b %d, %Y')}){trade_status}",
            f"<b>Conservative Entry</b> = {current_signal['conservative_entries'][0]:.2f}, {current_signal['conservative_entries'][1]:.2f}, {current_signal['conservative_entries'][2]:.2f}"
        ])
        
        # Targets section
        targets_text = []
        for i, (target, hit_date) in enumerate(zip(current_signal['targets'], current_signal['hit_dates'])):
            status = f"✅ HIT on {hit_date.strftime('%b %d')}" if hit_date else "⏳ PENDING"
            pct = format_pct_change(current_signal['entry'], target)
            targets_text.append(f"- TP {i+1} = {target:.2f} {pct} {status}")
        
        # Stop loss section
        sl_pct = format_pct_change(current_signal['entry'], current_signal['stop_loss'])
        stop_status = f"❌ HIT on {current_signal['stop_hit_date'].strftime('%b %d')}" if current_signal['hit_stop'] else "⏳ ACTIVE"
        table_content.extend(targets_text + ["", f"<b>Stop Loss</b> = {current_signal['stop_loss']:.2f} {sl_pct} {stop_status}"])
    else:
        table_content.append("No seller absorption trades found.")

    # Add summary table
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.03, y=0.97,
        text="<br>".join(table_content),
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0)",
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