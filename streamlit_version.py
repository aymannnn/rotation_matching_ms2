from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from copy import deepcopy

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Rotation Matcher", 
                   page_icon="🩺",
                   layout="wide")

st.title("The Medical Student Rotation Matcher")
st.caption(
    "Upload a CSV of student preferences and download the assigned rotations."
)


readme_text = Path("README_streamlit.md").read_text(encoding="utf-8")

with st.sidebar:
    with st.expander("📄 Instructions", expanded=True):
        st.markdown(readme_text)

# -----------------------------
# Rotation catalog & defaults (from your existing scripts)
# -----------------------------
# These names must match the columns in the uploaded CSV

ROTATION_NAMES = [
    'VA General Surgery',
    'VA VSU',
    'DRH General',
    'DRH VSU',
    'DRAH ACS',
    'DRAH Surgical Oncology',
    'DUH Surgical Oncology',
    'DUH Transplant',
    'DUH Colorectal',
    'DUH Pediatrics',
    'DUH Trauma/ACS',
    'DUH VSU',
    'DUH Breast/Endocrine',
    'DUH Cardiac',
    'DUH Thoracic',
    'DUH Melanoma',
]

# Location/subspecialty/default capacity for each rotation
ROTATION_META = [
    {"name": "VA General Surgery",    "location": "VA",
        "subspecialty": False, "default_max": 1},
    {"name": "VA VSU",                 "location": "VA",
        "subspecialty": True,  "default_max": 1},
    {"name": "DRH General",            "location": "DRH",
        "subspecialty": False, "default_max": 1},
    {"name": "DRH VSU",                "location": "DRH",
        "subspecialty": True,  "default_max": 1},
    {"name": "DRAH ACS",               "location": "DRAH",
        "subspecialty": False, "default_max": 2},
    {"name": "DRAH Surgical Oncology", "location": "DRAH",
        "subspecialty": False, "default_max": 1},
    {"name": "DUH Surgical Oncology",  "location": "DUH",
        "subspecialty": False, "default_max": 2},
    {"name": "DUH Transplant",         "location": "DUH",
        "subspecialty": True,  "default_max": 1},
    {"name": "DUH Colorectal",         "location": "DUH",
        "subspecialty": False, "default_max": 2},
    {"name": "DUH Pediatrics",         "location": "DUH",
        "subspecialty": False, "default_max": 1},
    {"name": "DUH Trauma/ACS",         "location": "DUH",
        "subspecialty": False, "default_max": 2},
    {"name": "DUH VSU",                "location": "DUH",
        "subspecialty": True,  "default_max": 3},
    {"name": "DUH Breast/Endocrine",   "location": "DUH",
        "subspecialty": True,  "default_max": 2},
    {"name": "DUH Cardiac",            "location": "DUH",
        "subspecialty": True,  "default_max": 2},
    {"name": "DUH Thoracic",           "location": "DUH",
        "subspecialty": True,  "default_max": 2},
    {"name": "DUH Melanoma",           "location": "DUH",
        "subspecialty": True,  "default_max": 2},
]

DEFAULT_CAPACITY = {m["name"]: m["default_max"] for m in ROTATION_META}
LOCATION_MAP = {m["name"]: m["location"] for m in ROTATION_META}
SUBSPEC_MAP = {m["name"]: m["subspecialty"] for m in ROTATION_META}

# -----------------------------
# Quick validator (top-level)
# -----------------------------


def _validate_preferences(df, rotation_names, max_row_errors=10):
    errors = []
    if 'Name' not in df.columns:
        errors.append("Missing required 'Name' column.")
        return errors
    if df['Name'].isna().any():
        errors.append("'Name' column contains missing values.")
    if df['Name'].duplicated().any():
        dups = df['Name'][df['Name'].duplicated()].unique()
        if len(dups) > 0:
            preview = ', '.join(map(str, dups[:5]))
            more = '...' if len(dups) > 5 else ''
            errors.append(f"Duplicate names found: {preview}{more}")

    present_rot_cols = [c for c in rotation_names if c in df.columns]
    if not present_rot_cols:
        errors.append("No recognized rotation columns found.")
        return errors

    M = len(present_rot_cols)
    sub = df[present_rot_cols].copy()
    for c in present_rot_cols:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    bad_rows = []
    needed = set(range(1, M+1))
    for idx, row in sub.iterrows():
        vals = row.values.astype('float')
        if np.isnan(vals).any():
            bad_rows.append((idx, 'NaNs in rankings'))
            continue
        ints = vals.astype(int)
        if not np.allclose(vals, ints):
            bad_rows.append((idx, 'Non-integer rankings'))
            continue
        if set(ints) != needed:
            bad_rows.append((idx, f"Ranks must be 1..{M} without repeats"))
        if len(bad_rows) >= max_row_errors:
            break
    if bad_rows:
        examples = "; ".join(
            [f"row {i}: {reason}" for i, reason in bad_rows[:max_row_errors]])
        errors.append(f"Ranking errors: {examples}")

    return errors


# -----------------------------
# Core algorithm (adapted to be parameterized + Streamlit friendly)
# -----------------------------
INDEX_BLOCK_ONE = 0
INDEX_BLOCK_TWO = 1


class Rotation:
    def __init__(self, name, location, maximum_students, subspecialty):
        self.name = name
        self.location = location
        self.maximum_students = maximum_students
        self.subspecialty = subspecialty
        self.block_one_count = 0
        self.block_two_count = 0
        self.full_block_one = False
        self.full_block_two = False
        self.full_rotation = False

    def add_specific_block(self, block):
        # Returns True if assignment succeeded, else False
        if block == INDEX_BLOCK_ONE and not self.full_block_one:
            self.block_one_count += 1
            self._update_full_flags()
            return True
        elif block == INDEX_BLOCK_TWO and not self.full_block_two:
            self.block_two_count += 1
            self._update_full_flags()
            return True
        return False

    def _update_full_flags(self):
        if self.block_one_count >= self.maximum_students:
            self.full_block_one = True
        if self.block_two_count >= self.maximum_students:
            self.full_block_two = True
        if self.full_block_one and self.full_block_two:
            self.full_rotation = True


def _calculate_row_cost(row, df_prefs):
    c1 = df_prefs.loc[df_prefs['Name'] ==
        row['Name']][row['Block One']].values[0]
    c2 = df_prefs.loc[df_prefs['Name'] ==
        row['Name']][row['Block Two']].values[0]
    return c1 + c2


def _calculate_solution_cost(assignments, df_prefs):
    assignments = assignments.copy()
    assignments['Cost'] = assignments.apply(
        _calculate_row_cost, args=(df_prefs,), axis=1)
    return assignments['Cost'].sum(), assignments


def _get_general_block(rotations):
    # Count open spots by block and type
    b1_general = b2_general = b1_sub = b2_sub = 0
    for r in rotations:
        if not r.subspecialty:
            b1_general += (r.maximum_students - r.block_one_count)
            b2_general += (r.maximum_students - r.block_two_count)
        else:
            b1_sub += (r.maximum_students - r.block_one_count)
            b2_sub += (r.maximum_students - r.block_two_count)

    if b1_general == 0 and b2_general == 0:
        return None  # no general capacity left

    if b1_general > 0 and b2_general > 0:
        # Prefer the block that leaves subspecialty flexibility
        if b1_sub > 0 and b2_sub > 0:
            return np.random.choice([INDEX_BLOCK_ONE, INDEX_BLOCK_TWO])
        elif b1_sub == 0:
            return INDEX_BLOCK_TWO
        elif b2_sub == 0:
            return INDEX_BLOCK_ONE

    if b1_general > 0 and b2_general == 0:
        return INDEX_BLOCK_ONE
    if b1_general == 0 and b2_general > 0:
        return INDEX_BLOCK_TWO

    return None


def _generate_solution(df_prefs, name_list, rot_defs):
    # Build Rotation objects (fresh each attempt)
    rotations = deepcopy(rot_defs)

    # Prepare output
    assignments = pd.DataFrame({
        'Name': name_list,
        'Block One': [None] * len(name_list),
        'Block Two': [None] * len(name_list),
    })

    # Work on a modifiable copy of preferences for dropping full rotations
    prefs_work = df_prefs.copy()

    for name in name_list:
        # Names of remaining general/subspecialty rotations
        GENERAL = [r.name for r in rotations if not r.subspecialty]
        SUBSPEC = [r.name for r in rotations if r.subspecialty]

        # Sort this student's general prefs ascending (1 = best)
        general_sorted = prefs_work.loc[prefs_work['Name']
            == name, GENERAL].iloc[0].sort_values()

        # Decide which block gets the general rotation
        g_block = _get_general_block(rotations)
        if g_block is None:
            return None  # infeasible
        if g_block == INDEX_BLOCK_TWO:
            g_col, s_col = 'Block Two', 'Block One'
            s_block = INDEX_BLOCK_ONE
        else:
            g_col, s_col = 'Block One', 'Block Two'
            s_block = INDEX_BLOCK_TWO

        # Place GENERAL
        choice_idx = 0
        while True:
            if choice_idx >= len(general_sorted):
                return None  # no workable general rotation for this student
            r_name = general_sorted.index[choice_idx]
            r_obj = next(r for r in rotations if r.name == r_name)
            ok = r_obj.add_specific_block(g_block)
            if ok:
                assignments.loc[assignments['Name'] == name, g_col] = r_name
                if r_obj.full_rotation:
                    rotations.remove(r_obj)
                    prefs_work.drop(columns=[r_name], inplace=True)
                break
            choice_idx += 1

        # Place SUBSPECIALTY (recompute list after potential drop)
        SUBSPEC = [r.name for r in rotations if r.subspecialty]
        subspec_sorted = prefs_work.loc[prefs_work['Name']
            == name, SUBSPEC].iloc[0].sort_values()

        choice_idx = 0
        while True:
            if choice_idx >= len(subspec_sorted):
                return None
            r_name = subspec_sorted.index[choice_idx]
            r_obj = next(r for r in rotations if r.name == r_name)
            ok = r_obj.add_specific_block(s_block)
            if ok:
                assignments.loc[assignments['Name'] == name, s_col] = r_name
                if r_obj.full_rotation:
                    rotations.remove(r_obj)
                    prefs_work.drop(columns=[r_name], inplace=True)
                break
            choice_idx += 1

    return assignments


def _simple_match(df_prefs, rot_defs, max_attempts=1000, rng_seed=None):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    names = df_prefs['Name'].unique()
    if len(names) != len(df_prefs):
        raise ValueError("Duplicate names detected; 'Name' must be unique.")

    best_cost = np.inf
    best_assign = None

    for _ in range(max_attempts):
        order = names.copy()
        np.random.shuffle(order)
        assign = _generate_solution(df_prefs, order, rot_defs)
        if assign is None:
            continue
        total_cost, _ = _calculate_solution_cost(assign, df_prefs)
        if total_cost < best_cost:
            best_cost = total_cost
            best_assign = assign

    if best_assign is None:
        raise RuntimeError(
            "No feasible assignment found. Try increasing capacities or verifying input.")

    return best_assign, int(best_cost)


def _build_rotations(capacity_by_name):
    # Build Rotation objects for algorithm
    rotations = []
    for meta in ROTATION_META:
        name = meta['name']
        rotations.append(
            Rotation(
                name=name,
                location=LOCATION_MAP[name],
                maximum_students=int(capacity_by_name.get(
                    name, meta['default_max'])),
                subspecialty=SUBSPEC_MAP[name],
            )
        )
    return rotations


# -----------------------------
# UI – CSV upload & capacity controls
# -----------------------------
with st.sidebar:
    st.header("Capacity by Rotation (per block)")
    capacities = {}
    for name in ROTATION_NAMES:
        capacities[name] = st.number_input(
            label=name,
            min_value=0,
            value=int(DEFAULT_CAPACITY[name]),
            step=1,
            help="Max students per block for this rotation",
        )

# -----------------------------
# Sample CSV generator (optional)
# -----------------------------
st.subheader("Generate Sample Data")
students_n = st.number_input(
    "Number of students", min_value=1, max_value=500, value=12, step=1)


def _generate_sample_df(n):
    data = {"Name": [f"Student {i+1}" for i in range(n)]}
    k = len(ROTATION_NAMES)
    for r in ROTATION_NAMES:
        data[r] = [None]*n
    for i in range(n):
        ranks = np.random.permutation(np.arange(1, k+1))
        for j, r in enumerate(ROTATION_NAMES):
            data[r][i] = int(ranks[j])
    return pd.DataFrame(data)


if st.button("Generate sample data"):
    sample_df = _generate_sample_df(int(students_n))
    st.dataframe(sample_df.head(20), use_container_width=True)
    st.download_button(
        label="Download sample CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_preferences.csv",
        mime="text/csv",
    )

# Algorithm attempts slider (main page)
attempts = st.slider("Algorithm attempts", min_value=100, max_value=3000, value=1000, step=100,
                     help="Higher attempts increase chance of lower cost, at the expense of longer wait times.")


st.markdown(
    """
Read the instructions in the sidebar before uploading your data.
"""
)

uploaded = st.file_uploader("Upload Student Data CSV", type=[
                            "csv"], accept_multiple_files=False)


if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Couldn't read CSV: {e}")
        st.stop()

    if 'Name' not in df.columns:
        st.error("CSV must include a 'Name' column.")
        st.stop()

    # Validate rotation columns
    missing_columns = [r for r in ROTATION_NAMES if r not in df.columns]
    if missing_columns:
        st.warning(
            "Some expected rotation columns are missing: " +
                ", ".join(missing_columns)
        )

    present_rot_cols = [c for c in ROTATION_NAMES if c in df.columns]
    if not present_rot_cols:
        st.error("No valid rotation columns found in the CSV.")
        st.stop()

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

# Run quick validator
errors = _validate_preferences(df, ROTATION_NAMES) if 'df' in locals() else []
if 'df' in locals():
    if errors:
        for e in errors:
            st.error(e)
        st.stop()
    else:
        st.success("Validation passed.")

    # Build Rotation objects with current capacities
    rotation_defs = _build_rotations(capacities)

    run = st.button("Run Matching")
    if run:
        try:
            assignments, cost = _simple_match(df, rotation_defs, max_attempts=attempts)
            st.success(f"Done! Total preference cost: {cost}")

            st.subheader("Assignments")
            st.dataframe(assignments, use_container_width=True)

            # Download CSV
            csv_bytes = assignments.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download assignments CSV",
                data=csv_bytes,
                file_name="final_assignments.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Upload a CSV to begin.")
