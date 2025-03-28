import pandas as pd
import re
from collections import defaultdict
import numpy as np
from scipy.stats import entropy

class DataPreprocesser:

    def __init__(self, datafilepath):
        self.df = pd.read_csv(datafilepath)
        self.text_matches= {} #dict of utterances and matched text | {[{},{}]}
        self.num_matches = {}
        self.metric_keys = ['timestamp', 'speaker', 'message', 'value', 'Case Match Type', 'matchidx', 'convolen', 'uttidx', 'matchfreq', 'speaker_id']
        self.match_stats = {}
        '''{{phrase_to_match: dataframe}, }'''
        self.text_matches_new = {} 
        self.utterancesDF = None

    '''Match has: row_id, timestamp, speaker, message, value, utt_idx, speaker_id, case_match_type, match_idx'''
    '''Matched Convo has: match_freq, convo_len'''

    def getMatchedUtterancesDF(self, key_val):
        # Make sure both DataFrames have the same index for comparison
        match_idx_df = self.text_matches_new[key_val][0]
        if len(self.utterancesDF) != len(match_idx_df):
            raise ValueError("DataFrames must have the same number of rows for comparison.")

        # Filter the rows where 'match_idx' is True in the match_idx column of the second DataFrame
        matched_indices = match_idx_df[match_idx_df['match_idx'] == True].index

        # Now, filter self.utterancesDF using these matched indices
        matched_utterances = self.utterancesDF.loc[matched_indices]

        return matched_utterances
    
    def getMatchedConvoDF(self, key_val):
        match_idx_df = self.text_matches_new[key_val][1]  # This contains 'row_idx' and 'matchfreq'
        df_main = self.getDataframe()  # Main utterances dataframe
        # Find unique 'row_idx' values where 'matchfreq' is nonzero
        matched_row_idxs = match_idx_df.loc[match_idx_df['match_freq'] != 0, 'row_idx'].unique()
        # Filter self.getDataframe() where 'row_idx' is in matched_row_idxs
        matched_convos = df_main[df_main.index.isin(matched_row_idxs)]
        return matched_convos

    def checkAI(self, row_idx):
        return self.df["is_AI"].iloc[row_idx]

    # def process_speaker_counts(self, key_list):
        to_ret = []
        for key_val in key_list:
            match_dict_by_idx = self.text_matches[key_val]  # This is a dict with lists of dicts
            speaker_counts = defaultdict(lambda: {"Buyer": 0, "Seller": 0})

            # Iterate over the list of dictionaries, not the key itself
            for entry_list in match_dict_by_idx.values():  
                for entry in entry_list:  # Each entry is a dictionary
                    if isinstance(entry, dict):  # Ensure it's a dictionary before processing
                        case_type = entry.get("Case Match Type", "Unknown")
                        speaker = entry.get("speaker", "Unknown")
                        if speaker in ["Buyer", "Seller"]:
                            speaker_counts[case_type][speaker] += 1  

            to_ret.append({key_val: dict(speaker_counts)})

        return to_ret  # Convert defaultdict to regular dict
   
    def parsedtoDF(self):
        all_rows = []
        for row_idx, parsed_dialog in enumerate(self.df["parsed_dialog"]):
            for entry in parsed_dialog:
                entry["row_idx"] = row_idx
                entry["match_idx"] = False  # Boolean column
                entry["Case Match Type"] = None  # Empty string
                all_rows.append(entry)

        
        self.utterancesDF = pd.DataFrame(all_rows)
        self.utterancesDF["Case Match Type"] = self.utterancesDF["Case Match Type"].astype(object)
        # Compute conversation length per row_idx
        convolen_ut = self.utterancesDF.groupby("row_idx").size().rename("convo_len")
        # Assign conversation length to each utterance in df
        self.utterancesDF = self.utterancesDF.merge(convolen_ut, on="row_idx", how="right")

    def getUtterancesDF(self):
        return self.utterancesDF
   
    def parseRow(self, row_idx, row_entry, col_name):
        structured_dialog = []
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
                        'value': row_entry,
                        'uttidx': None,
                        'speaker_id': spk+ '_'+str(row_idx)
                            })
            return
        pattern = re.compile(r'^\s*(\d+|nan)?\s*(Buyer|Seller):\s*(.*)$', re.IGNORECASE)
        
        line_count =0
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
                    'value': None,
                    'uttidx': line_count,
                    'speaker_id': speaker + '_' + str(row_idx) if speaker is not None else None

                })
                line_count +=1
            else:
                # TODO: Handle AI Chats
                if structured_dialog and not line.startswith("Submitted agreement:"):
                    structured_dialog[-1]['message'] += " " + line
                else:
                # some other text response for self-report or survery by speaker 
                    structured_dialog.append({
                        'timestamp': None,
                        'speaker': spk,
                        'message': line,
                        'value': None,
                        'uttidx': line_count,
                        'speaker_id': spk + '_' + str(row_idx) if spk is not None else None

                            })
                    line_count +=1
        return structured_dialog

    def addParsedTextColumn(self, col_name, col_to_add):
        # Convert all NaN to empty strings right in the DataFrame
        self.dropChatNA()
        print(len(self.df))
        #self.df[col_name] = self.df[col_name].fillna("")
        """
        Adds a new column 'parsed_dialog' to the DataFrame containing structured dialog data.
        """
        # Ensure the 'formattedChat' column exists
        if col_name not in self.df.columns:
            raise ValueError("DataFrame must contain 'col_name' column.")
        # Apply the formatChat function to each row and create a new column
        parsed_rows = []
        for index, row in self.df.iterrows():
            row_value = row[col_name]
            parsed_row = self.parseRow(index, row_value, col_name)
            parsed_rows.append(parsed_row)
        self.df[col_to_add] = parsed_rows
        self.parsedtoDF()
   

    ''' Functions for matched key words'''
    def filterMatches(self, col_name, value_to_check):
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
        # Ensure the column is string type, replacing NaN values with empty strings
        df = self.utterancesDF

        df[col_name] = df[col_name].fillna("").astype(str)

        # Create Boolean masks for each match type
        exact_match = df[col_name].str.contains(value_to_check, na=False)
        lower_match = df[col_name].str.contains(r'\b' + re.escape(value_to_check.lower()) + r'\b', na=False) & ~exact_match
        case_insensitive_match = df[col_name].str.contains(r'\b' + re.escape(value_to_check) + r'\b', case=False, na=False) & ~exact_match & ~lower_match

        # Assign match types
        #df["Case Match Type"] = None  # Initialize column
        df.loc[exact_match, "Case Match Type"] = "Exact"
        df.loc[lower_match, "Case Match Type"] = "Lower"
        df.loc[case_insensitive_match, "Case Match Type"] = "Case Insensitive"

        df.loc[exact_match | lower_match | case_insensitive_match, "match_idx"] = True
        

        # Filter matched utterances (keep same length)
        matched_df_utt = df[["row_idx", "match_idx", "Case Match Type", "convo_len"]]

        #row_matched = df[df["match_idx"] == True][["row_idx", "match_idx", "Case Match Type", "convo_len"]]
        # Copute match frequency (count matches per row_idx group)
        match_freq = df.groupby("row_idx")['Case Match Type'].apply(lambda x: x.isin(["Exact", "Lower", "Case Insensitive"]).sum()).reset_index(name="match_freq")
        # Compute conversation length per row_idx
        convolen = df.groupby("row_idx").size().reset_index(name='convo_len')
        # Merge match_freq and convolen on row_idx to create the summary DataFrame
        convo_stats = pd.merge(match_freq, convolen, on="row_idx", how="left")   
        convo_stats.head() 
        utt_stats = pd.DataFrame()
        self.text_matches_new[value_to_check] = [matched_df_utt, convo_stats, utt_stats]
        self.normalizedRelativePos(value_to_check)

    def getConvoMatchesByCase(self, value_key):
        df_utt = self.utterancesDF
        df_utt['Case Match Type'] = self.text_matches_new[value_key][0]['Case Match Type']
        # df_utt.groupby("row_idx")['Case Match Type'].apply(lambda x: x.isin(["Exact", "Lower", "Case Insensitive"]).sum())
        group = df_utt.groupby("row_idx")["Case Match Type"].value_counts().unstack(fill_value=0)
        df_utt['Case Match Type'] = np.nan
        df_utt['Case Match Type'] = df_utt['Case Match Type'].astype('object')
        return group

    def groupbyMatchUttStat(self, value_key, group_by, stat_col, agg_list):
        df_utt = self.getMatchedUtterancesDF(value_key)
        df_utt[stat_col] = self.text_matches_new[value_key][2][stat_col] # Assuming this exists
        stat = df_utt.groupby(group_by)[stat_col].agg(agg_list)
        return stat
    
    def groupbyMatchConvoStat(self, value_key, group_by, stat_col, agg_list):
        df_convo = self.getMatchedConvoDF(value_key)
        df_convo[stat_col] = self.text_matches_new[value_key][1][stat_col]  # Assuming this exists
        stat = df_convo.groupby(group_by)[stat_col].agg(agg_list)
        return stat
    
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
    
    def dropChatNA(self):
        self.df = self.df.dropna(subset=["formattedChat"]).reset_index(drop=True)

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

        
    ''' All statistics for words'''
    # def speakerPhrase(self):
    def getTTR(self, key_value):
        df = self.text_matches_new[key_value][1]
        convos_with_phrase = df['match_freq'].sum()
      
        return pd.to_numeric(convos_with_phrase/ (df.shape[0]))
        
    def getDispersion(self, key_val):
        #f_i  = frequency of phrase in conversation 
        #f_bar mean frequency across all conversations.
        df = self.text_matches_new[key_val][1]
        f_mean  = df['match_freq'].mean()
        dispersion_index = ((df['match_freq'] - f_mean) ** 2).sum() / f_mean
        return dispersion_index

    def getEntropy(self, key_val):
        df = self.text_matches_new[key_val][1]
        phrase_counts = df['match_freq']
        phrase_probs = phrase_counts / phrase_counts.sum()
        phrase_entropy = entropy(phrase_probs, base=2)
        return phrase_entropy

    def getStandardizedDispersion(self, key_val):
        df = self.text_matches_new[key_val][1]
        f_mean  = df['match_freq'].mean()
        std_dev = df['match_freq'].std()
        num_conversations = len(df['match_freq'])
        juilland_d = 1 - (std_dev / (f_mean * np.sqrt(num_conversations)))
        return juilland_d

    def normalizedRelativePos(self, key_value):
        #save this column to getMatchedUtterancesDF
        df = self.getMatchedUtterancesDF(key_value)
        df_2 = self.text_matches_new[key_value][2] #stats_df for matched cases
        df_2['relative_pos'] = df["uttidx"] / df["convo_len"].replace(1, float('nan'))
        df_2['relative_pos'] = pd.to_numeric(df["uttidx"] / (df["convo_len"] - 1))
        # self.text_matches_new[key_value][2] = df_2
 
    def getSD(self, col_name, key_value):
        df_col = self.getMatchedUtterancesDF['col name']
        return np.std(df_col)



if __name__ == "__main__":
        filepath = "/Users/mishkin/Desktop/Research/Convo_Kit/ConvoKit_Disputes/data/alldyads.csv"
        data_preprocessor = DataPreprocesser(filepath)
        #data_preprocessor.addParsedDialogColumn()
        data_preprocessor.show()

  # def matchedWordtoDF(self, matched_dict, key_val):
        # data = []
        # convodata = []
        # # print(matched_dict)
        # for row_id, entries in matched_dict.items():
        #     for entry in entries:

        #         if 'message' in entry:  # Conversation details
        #             data.append({
        #                 'row_id': row_id,
        #                 'timestamp': entry.get('timestamp'),
        #                 'speaker': entry.get('speaker'),
        #                 'message': entry.get('message'),
        #                 'value': entry.get('value'),
        #                 'uttidx': entry.get('uttidx'),
        #                 'speaker_id': entry.get('speaker_id'),
        #                 'Case Match Type': entry.get('Case Match Type'),
        #                 # 'matchidx': entry.get('matchidx'),
        #             })
        #         else:  # Metadata details
        #             convodata.append({
        #                 'row_id': row_id,
        #                 'convolen': entry.get('convolen'),
        #                 'matchfreq': entry.get('matchfreq')
        #             })
        # df_conversations = pd.DataFrame(data)
        # df_convodata = pd.DataFrame(convodata)
        # self.text_matches_new[key_val] = [df_conversations, df_convodata]
'''
    if not (isinstance(value_to_check, (str))):
            self.num_matches.append({})
        
        match_dict = {}
        for row_idx in row_indices:
            row_value = self.df.at[row_idx, col_name]
            if isinstance(row_value , list):  
                match_freq=0
                matches =[]
                for entry in self.df[col_name].iloc[row_idx]:
                    copy_entry = entry.copy() 
                    if isinstance(entry, dict):  
                        message_value = entry["message"]
                        if message_value and isinstance(message_value, str) and value_to_check in message_value:
                            match_freq +=1
                            copy_entry.update({"matchidx":entry.get("uttidx", "Unknown")})
                            copy_entry.update({"Case Match Type":"Exact"})
                            matches.append(copy_entry)
                        elif message_value and isinstance(message_value, str) and value_to_check.lower() in message_value:
                            match_freq +=1
                            copy_entry.update({"matchidx":entry.get("uttidx", "Unknown")})
                            copy_entry.update({"Case Match Type":"Lower"})
                            matches.append(copy_entry)
                        elif message_value and isinstance(message_value, str) and value_to_check.lower() in message_value.lower():
                            match_freq +=1
                            copy_entry.update({"matchidx":entry.get("uttidx", "Unknown")})
                            copy_entry.update({"Case Match Type":"Case Insensitive"})
                            matches.append(copy_entry)
                        else:
                            copy_entry.update({"Case Match Type":"No Match"})
                #at end of each row, add {'convolen': len(entry), 'matchfreq': match_freq}
            matches.append({'convolen': len(row_value), 'matchfreq': match_freq})
            match_dict[str(row_idx)] = matches
        
        self.text_matches[str(value_to_check)] = match_dict
        self.matchedWordtoDF(match_dict, value_to_check)
        return match_dict
    '''  

