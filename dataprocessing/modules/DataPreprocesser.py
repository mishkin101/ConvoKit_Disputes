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

if __name__ == "__main__":
    filepath = "/Users/mishkin/Desktop/Research/Convo_Kit/ConvoKit_Disputes/data/alldyads.csv"
    data_preprocessor = DataPreprocesser(filepath)
    #data_preprocessor.addParsedDialogColumn()
    data_preprocessor.show()




