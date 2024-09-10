# come back to later 

import pandas as pd

def daily_leads_and_partners(daily_sales: pd.DataFrame) -> pd.DataFrame:
    unique_sales = daily_sales.drop_duplicates()
    
    return unique_sales
