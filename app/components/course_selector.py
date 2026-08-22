"""
Course selection component.

Renders an interactive AgGrid table and returns the courses the user picked.

Bug fixes applied:
  - Bug #10: pd.DataFrame(selected_rows) is called WITHOUT passing columns=,
             which previously produced an all-NaN frame because AgGrid returns
             a list of dicts (not a list of lists).
"""
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder


def render_course_selector(course_df: pd.DataFrame) -> pd.DataFrame:
    """
    Display a multi-select course table.

    Parameters
    ----------
    course_df : pd.DataFrame
        Must contain at least COURSE_ID, TITLE, and DESCRIPTION columns.

    Returns
    -------
    pd.DataFrame
        Selected rows with columns [COURSE_ID, TITLE].
        Empty DataFrame if nothing is selected.
    """
    st.subheader("Select courses that you have audited or completed:")

    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        theme="streamlit",
        fit_columns_on_grid_load=True,
        height=400,
    )

    # Bug #10 fix: AgGrid returns list-of-dicts; construct DataFrame without columns= arg
    selected_rows = response.get("selected_rows") or []
    if not selected_rows:
        return pd.DataFrame(columns=["COURSE_ID", "TITLE"])

    selected_df = pd.DataFrame(selected_rows)
    if "COURSE_ID" in selected_df.columns and "TITLE" in selected_df.columns:
        return selected_df[["COURSE_ID", "TITLE"]].reset_index(drop=True)

    return pd.DataFrame(columns=["COURSE_ID", "TITLE"])
