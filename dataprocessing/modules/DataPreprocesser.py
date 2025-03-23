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

    def addParsedDialogColumn(self):
        # Convert all NaN to empty strings right in the DataFrame
        self.df['formattedChat'] = self.df['formattedChat'].fillna("")
        """
        Adds a new column 'parsed_dialog' to the DataFrame containing structured dialog data.
        """
        # Ensure the 'formattedChat' column exists
        if 'formattedChat' not in self.df.columns:
            raise ValueError("DataFrame must contain 'formattedChat' column.")
        # Apply the formatChat function to each row and create a new column
        parsed_dialogs = []
        for _, row in self.df.iterrows():
            chat_text = row['formattedChat']
            dialog_list = self.parseChat(chat_text)
            parsed_dialogs.append(dialog_list)
        self.df['parsed_dialog'] = parsed_dialogs

    def show(self):
        """
        Displays the first few rows of the DataFrame for inspection.
        """
        self.df.head()

    def checkSubstring(self, col_name, row_idx, string_to_check):
        """
        Checks if the phrase 'Walk Away' appears in any value within the 'parsed_dialog' dictionary.

        Parameters:
        -----------
        parsed_dialog : list of dict
            A list of dictionaries representing structured dialogue.

        Returns:
        --------
        bool
            True if 'Walk Away' is found in any dictionary value, False otherwise.
        """
        if not isinstance(self.df[col_name].iloc[row_idx], list):  
            
            return False  # Ensure it's a list before processing

        for entry in self.df[col_name].iloc[row_idx]:
            if isinstance(entry, dict):  
                message_value = entry["message"]
                if message_value and isinstance(message_value, str) and string_to_check.lower() in message_value:
                    print("Lower case match:")
                    return entry
                elif message_value and isinstance(message_value, str) and string_to_check in message_value:
                    print("Exact case match:")
                    return entry
                elif message_value and isinstance(message_value, str) and string_to_check.lower() in message_value.lower():
                    print("No case match:")
                    return entry
            return False
    
    # def filterCase(self, val_to_filter, col_idx):

    # def checkDiff(self,val_to_filter, col_idx):

    def getDataframe(self):
        """
        Returns the DataFrame object.
        """
        return self.df
if __name__ == "__main__":
    filepath = "/Users/mishkin/Desktop/Research/Convo_Kit/ConvoKit_Disputes/data/alldyads.csv"
    data_preprocessor = DataPreprocesser(filepath)
    #data_preprocessor.addParsedDialogColumn()
    data_preprocessor.show()




