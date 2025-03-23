import pandas as pd
import re

class DataPreprocesser:

    def __init__(self, datafilepath):
        self.df = pd.read_csv(datafilepath) #dataframe object

    def parseChat(self, chat_text):
        # Convert to string to avoid errors (NaN -> 'nan' -> we'll treat that like empty)
        if pd.isnull(chat_text):
            chat_text = ""  # or "No chat available"
        
        pattern = re.compile(r'^\s*(\d+|nan)?\s*(Buyer|Seller):\s*(.*)$', re.IGNORECASE)
        structured_dialog = []
        
        for line in str(chat_text).split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = pattern.match(line)
            if match:
                timestamp_str, speaker, message = match.groups()
                timestamp = int(timestamp_str) if timestamp_str.isdigit() else timestamp_str
                
                structured_dialog.append({
                    'timestamp': timestamp,
                    'speaker': speaker,
                    'message': message.strip()
                })
            '''
            else:
                if structured_dialog and not line.startswith("Submitted agreement:"):
                    structured_dialog[-1]['message'] += " " + line
                else:
                    structured_dialog.append({
                        'timestamp': None,
                        'speaker': None,
                        'message': line
                            })
            '''
      
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
        parsed_dialogs = []
        for _, row in self.df.iterrows():
            chat_text = row[col_name]
            dialog_list = self.parseChat(chat_text)
            parsed_dialogs.append(dialog_list)
        self.df[col_to_add] = parsed_dialogs

    def show(self):
        """
        Displays the first few rows of the DataFrame for inspection.
        """
        self.df.head()

    def filterTextMatches(self, col_name, row_indices, string_to_check):
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
        match_dict = {}
        for row_idx in row_indices:
            if not isinstance(self.df[col_name].iloc[row_idx], list):  
                return False  # Ensure it's a list before processing
            matches = []
            for entry in self.df[col_name].iloc[row_idx]:
                if isinstance(entry, dict):  
                    message_value = entry["message"]
                    if message_value and isinstance(message_value, str) and string_to_check.lower() in message_value:
                        print("Lower case match:")
                        matches.append(entry)
                    elif message_value and isinstance(message_value, str) and string_to_check in message_value:
                        print("Exact case match:")
                        matches.append(entry)
                    elif message_value and isinstance(message_value, str) and string_to_check.lower() in message_value.lower():
                        print("No case match:")
                        matches.append(entry)
            match_dict[row_idx] = matches
        return match_dict
        
    def filterExcludeRows(self, column, include_str, exclude_str, case_1, case_2):
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
        
        filtered_df = self.df[self.df[column].str.contains(include_str, case=case_1, na=False)]
        
        final_df = filtered_df[~filtered_df[column].str.contains(exclude_str, case=case_2, na=False)]
        
        return final_df

    def getDataframe(self):
        """
        Returns the DataFrame object.
        """
        return self.df
    
    def getIdxMatch(self,idx_list):
        """
        Checks if the indices in idx_list exist in the DataFrame and returns the corresponding rows.
        """
        return self.df.loc[self.df.index.intersection(idx_list)]






if __name__ == "__main__":
    filepath = "/Users/mishkin/Desktop/Research/Convo_Kit/ConvoKit_Disputes/data/alldyads.csv"
    data_preprocessor = DataPreprocesser(filepath)
    #data_preprocessor.addParsedDialogColumn()
    data_preprocessor.show()




