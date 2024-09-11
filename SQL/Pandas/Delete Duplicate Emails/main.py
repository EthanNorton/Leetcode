# hard, coming back 

import pandas as pd

def delete_duplicate_emails(person: pd.DataFrame) -> None:
    result = person.remove()

    return result['id','email']