# Streamlit touch-up CSS — deep-navy sidebar, crisp light body, modern card treatment.
CSS = """
<style>
  /* global typography — Inter for headings, system UI for body */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  h1, h2, h3, h4, h5, h6 { font-family: 'Inter', system-ui, sans-serif; font-weight: 700; }
  .block-container { padding-top: 2.4rem; padding-bottom: 5rem; max-width: 1280px; }

  /* ─── kicker / eyebrow label ─── */
  .kicker {
    font-size: 0.75rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: #5c6abf; font-weight: 700; margin: 0 0 0.4rem;
  }
  .subtitle {
    color: #4a4a5a; max-width: 72ch; font-size: 1.02rem; line-height: 1.6; font-weight: 400;
  }

  /* ─── headline metrics ─── */
  div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e2e4f0; border-radius: 16px;
    padding: 1.1rem 1.2rem 0.8rem;
    box-shadow: 0 1px 3px rgba(42,44,72,.06), 0 4px 14px rgba(42,44,72,.08);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
  }
  div[data-testid="stMetric"]:hover {
    box-shadow: 0 2px 6px rgba(42,44,72,.1), 0 8px 24px rgba(42,44,72,.12);
    transform: translateY(-1px);
  }
  div[data-testid="stMetricLabel"] {
    font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase;
    color: #7b7f9e; font-weight: 600;
  }
  div[data-testid="stMetricValue"] {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.025em;
    font-variant-numeric: tabular-nums; color: #2a2c48;
  }
  div[data-testid="stMetricValue"] .st-emotion-cache-0 {
    font-size: 0.95rem; font-weight: 600; color: #7b7f9e;
  }

  /* ─── sub-headers ─── */
  h3, [data-testid="stSubheader"] {
    font-size: 1.25rem; font-weight: 700; color: #2a2c48;
    margin-top: 2.2rem; margin-bottom: 0.4rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #f0f1f8;
  }

  /* ─── section captions ─── */
  .stCaption { font-size: 0.85rem; color: #6b6e88; margin-top: -0.2rem; margin-bottom: 1rem; font-weight: 400; }

  /* ─── miss cards ─── */
  .misscard {
    border: 1px solid #e8dfe8; border-left: 4px solid #e06060;
    background: #ffffff; border-radius: 14px; padding: 1rem 1.1rem 0.9rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 4px rgba(42,44,72,.05);
  }
  .misscard:hover { box-shadow: 0 3px 10px rgba(42,44,72,.1); }
  .misscard .mr { font-weight: 800; font-size: 1.1rem; color: #2a2c48; margin-right: 0.5rem; }
  .misscard .mt { font-weight: 600; color: #2a2c48; font-size: 0.95rem; margin-top: 0.3rem; }
  .misscard .mx { color: #4a4a5a; font-size: 0.88rem; margin-top: 0.25rem; line-height: 1.55; }
  .misscard .chip {
    display:inline-block; font-size: 0.68rem; font-weight: 700;
    border-radius: 999px; padding: 0.2rem 0.65rem; margin-right: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;
  }
  .chip.neg { background:#fee4e4; color:#d4373b; }
  .chip.pos { background:#dcecd8; color:#22863a; }
  .chip.neu { background:#e4e8f5; color:#4a5abf; }

  /* ─── info / callout boxes ─── */
  [data-testid="stAlert"] {
    border-radius: 14px; border: 1px solid #e2e4f0;
    background: #f8f9fd;
  }
  [data-testid="stAlert"] p { font-size: 0.92rem; color: #4a4a5a; line-height: 1.6; }

  /* ─── data table ─── */
  [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
  [data-testid="stDataFrame"] tbody tr { transition: background 0.15s ease; }
  [data-testid="stDataFrame"] tbody tr:hover { background: #f4f5fa !important; }

  /* ─── sidebar ─── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e3a 0%, #2a2c48 100%);
  }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] [data-testid="stSubheader"] {
    color: #ffffff; font-size: 1rem; font-weight: 700; border-bottom: none; margin-top: 0.5rem;
  }
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] div,
  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #c5c7d6 !important; font-size: 0.88rem; line-height: 1.6;
  }
  section[data-testid="stSidebar"] [data-testid="stCaption"] { color: #8b8da8 !important; font-size: 0.78rem; }
  section[data-testid="stSidebar"] .stButton button {
    background: #ffffff; color: #2a2c48; font-weight: 600; border-radius: 8px;
    border: none; padding: 0.5rem 1rem; font-size: 0.85rem; width: 100%;
    transition: background 0.2s ease;
  }
  section[data-testid="stSidebar"] .stButton button:hover { background: #e8e9f4; }
  section[data-testid="stSidebar"] hr { border-color: #3d3f5c; }
  section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button {
    background: #3d3f5c !important; color: #c5c7d6 !important; border-radius: 8px;
  }
  section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    background: #5c6abf !important; color: #ffffff !important;
  }
  section[data-testid="stSidebar"] select select { color: #2a2c48; }
  section[data-testid="stSidebar"] select { background: #ffffff; border-radius: 8px; }

  /* ─── main divider ─── */
  [data-testid="stDivider"] hr { border-color: #e2e4f0; }

  /* ─── select boxes & inputs ─── */
  .stSelectbox label, .stTextInput label { color: #4a4a5a; font-weight: 600; font-size: 0.82rem; letter-spacing: 0.03em; }
  .stSelectbox div[data-baseweb="select"], .stTextInput div[data-baseweb="input"] {
    border-radius: 8px; border-color: #d4d6e4;
  }

  /* ─── footer ─── */
  footer { margin-top: 2rem; }
  footer p { font-size: 0.78rem; color: #8b8da8; }
</style>
"""