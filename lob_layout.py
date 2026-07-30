"""Column layout of the 40 FI-2010 LOB features.

Standard FI-2010 ordering (Ntakaris et al.): the 40 features are 10 price
levels, each level laid out as ``[P_ask, V_ask, P_bid, V_bid]``. So for level
``i`` (0-indexed) the four columns live at ``4*i .. 4*i+3``.

This assumed layout is VERIFIED empirically in ``data_analysis/eda.py`` (real
DecPre data should show ask prices ascending and bid prices descending across
levels) before it is relied on for the validity penalty or metrics.
"""

N_LEVELS = 10
N_LOB_FEATURES = 40

ASK_PRICE_IDX = [4 * i + 0 for i in range(N_LEVELS)]  # 0, 4, ..., 36
ASK_VOL_IDX = [4 * i + 1 for i in range(N_LEVELS)]     # 1, 5, ..., 37
BID_PRICE_IDX = [4 * i + 2 for i in range(N_LEVELS)]   # 2, 6, ..., 38
BID_VOL_IDX = [4 * i + 3 for i in range(N_LEVELS)]     # 3, 7, ..., 39

# Level-1 (best) quotes.
BEST_ASK_PRICE = ASK_PRICE_IDX[0]
BEST_BID_PRICE = BID_PRICE_IDX[0]
BEST_ASK_VOL = ASK_VOL_IDX[0]
BEST_BID_VOL = BID_VOL_IDX[0]
