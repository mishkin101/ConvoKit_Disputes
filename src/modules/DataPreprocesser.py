import pandas as pd
import re
from collections import defaultdict

class DataPreprocesser:

    def __init__(self, datafilepath):
        self.df = pd.read_csv(datafilepath)
        self.text_matches= {} #dict of utterances and matched text
        self.num_matches = {}

    def parseRow(self, row_idx, row_entry, col_name):
        structured_dialog = []
        # Convert to string to avoid errors (NaN -> 'nan' -> we'll treat that like empty)
        if pd.isnull(row_entry):
            row_entry = ""  # or "No chat available"
        
        spk =self.getSpeakerFromCol(col_name, row_idx)
        if isinstance(row_entry, (int,float)):
            structured_dialog.append({
                        'timestamp': None,
                        'speaker': spk,
                        'message': None,
                        'value': row_entry
                            })
            return
        pattern = re.compile(r'^\s*(\d+|nan)?\s*(Buyer|Seller):\s*(.*)$', re.IGNORECASE)
        
        for line in str(row_entry).split('\n'):
            line = line.strip()
            # get rid of empty text
            if not line:
                continue
            
            match = pattern.match(line)
            if match:
                timestamp_str, speaker, message = match.groups()
                timestamp = int(timestamp_str) if timestamp_str.isdigit() else timestamp_str
                
                structured_dialog.append({
                    'timestamp': timestamp,
                    'speaker': speaker,
                    'message': message.strip(),
                    'value': None
                })
            
            else:
                if structured_dialog and not line.startswith("Submitted agreement:"):
                    structured_dialog[-1]['message'] += " " + line
                else:
                    structured_dialog.append({
                        'timestamp': None,
                        'speaker': spk,
                        'message': line,
                        'value': None
                            })
        
      
        return structured_dialog

    def addParsedTextColumn(self, col_name, col_to_add):
        # Convert all NaN to empty strings right in the DataFrame
        self.df[col_name] = self.df[col_name].fillna("")
        """
        Adds a new column 'parsed_dialog' to the DataFrame containing structured dialog data.
        """
        # Ensure the 'formattedChat' column exists
        if col_name not in self.df.columns:
            raise ValueError("DataFrame must contain 'formattedChat' column.")
        # Apply the formatChat function to each row and create a new column
        parsed_Rows = []
        for index, row in self.df.iterrows():
            row_value = row[col_name]
            dialog_list = self.parseRow(index,row_value, col_name)
            parsed_Rows.append(dialog_list)
        self.df[col_to_add] = parsed_Rows

    def show(self):
        """
        Displays the first few rows of the DataFrame for inspection.
        """
        self.df.head()

    def filterMatches(self, col_name, row_indices, value_to_check):
        """
        Checks if the phrase 'string_to_check' appears in any value within the text for 'col_name' dictionary.

        Parameters:
        -----------
        parsed_dialog : list of dict
            A list of dictionaries representing structured dialogue.

        Returns:
        --------
        bool
            True if 'Walk Away' is found in any dictionary value, False otherwise.
        """
        #add later
        if not (isinstance(value_to_check, (str))):
            self.num_matches.append({})
        match_dict = {}
        for row_idx in row_indices:
            row_value = self.df.at[row_idx, col_name]
            if isinstance(row_value , list):  
                matches = []
                for entry in self.df[col_name].iloc[row_idx]:
                    copy_entry = entry.copy() 
                    if isinstance(entry, dict):  
                        message_value = entry["message"]
                        if message_value and isinstance(message_value, str) and value_to_check in message_value:
                            copy_entry.update({"Case Match Type":"Exact"})
                            matches.append(copy_entry)
                        if message_value and isinstance(message_value, str) and value_to_check.lower() in message_value:
                            copy_entry.update({"Case Match Type":"Lower"})
                            matches.append(copy_entry)
                        elif message_value and isinstance(message_value, str) and value_to_check.lower() in message_value.lower():
                            copy_entry.update({"Case Match Type":"Case Insensitive"})
                            matches.append(copy_entry)
                        else:
                            copy_entry.update({"Case Match Type":"None"})

            match_dict[row_idx] = matches  

        self.text_matches[str(value_to_check)] = match_dict
        return match_dict
        
    def filterRows(self, column, include_val=None , exclude_val=None, case_in=None, case_ex=None):
        """
        Filters the DataFrame to include rows where the specified column contains 'include_str'
        and excludes rows where it contains 'exclude_str'.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to filter.
        column : str
            The name of the column to apply the filter on.
        include_str : str
            The string that must be included in the column value.
        exclude_str : str
            The string that must not be included in the column value.

        Returns:
        --------
        pandas.DataFrame
            The filtered DataFrame.
        """
    
        filtered_df = self.df  # Start with the full DataFrame
    
        # Handle inclusion filtering
        if include_val is not None:
            if isinstance(include_val, (int, float)):  # Numeric case
                filtered_df = filtered_df[filtered_df[column] == include_val]
            else:  # String case
                filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(include_val), case=case_in, na=False)]
        
        # Handle exclusion filtering
        if exclude_val is not None:
            if isinstance(exclude_val, (int, float)):  # Numeric case
                filtered_df = filtered_df[filtered_df[column] != exclude_val]
            else:  # String case
                filtered_df = filtered_df[~filtered_df[column].astype(str).str.contains(str(exclude_val), case=case_ex, na=False)]

        return filtered_df

    def getDataframe(self):
        """
        Returns the DataFrame object.
        """
        return self.df
    
    def filterIdxMatch(self,idx_list):
        """
        Checks if the indices in idx_list exist in the DataFrame and returns the corresponding rows.
        """
        return self.df.loc[self.df.index.intersection(idx_list)]

    def getSpeakerFromCol(self, col_name, row_idx):
            if self.checkAI(row_idx):
                spk = "AI"
            else:
                if col_name.lower().startswith("b"):
                    spk = "Buyer"
                elif col_name.lower().startswith("s"):
                    spk = "Seller"
                else:
                    spk = None
            return spk
    
    def checkAI(self, row_idx):
        return self.df["is_AI"].iloc[row_idx]
    # def aggregateMatches(self, match_list, key_list, operation):

    def aggregrateMatches(self, key_list):
        to_ret = []
        
        for key_val in key_list:
            match_dict_by_idx = self.text_matches[key_val]  
            speaker_counts = defaultdict(lambda: {"Buyer": 0, "Seller": 0})


            for entry_list in match_dict_by_idx.values():  
                for entry in entry_list: 
                    if isinstance(entry, dict):  
                        case_type = entry.get("Case Match Type", "Unknown")
                        speaker = entry.get("speaker", "Unknown")
                        if speaker in ["Buyer", "Seller"]:
                            speaker_counts[case_type][speaker] += 1  
            to_ret.append({key_val: dict(speaker_counts)})
        return to_ret  


        


if __name__ == "__main__":
        filepath = "/Users/mishkin/Desktop/Research/Convo_Kit/ConvoKit_Disputes/data/alldyads.csv"
        data_preprocessor = DataPreprocesser(filepath)
        #data_preprocessor.addParsedDialogColumn()
        data_preprocessor.show()




