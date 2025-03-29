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
        #df[["Case Match Type", "is_AI"]] = df[["Case Match Type", "is_AI"]].values

        return matched_utterances
    
    def getMatchedConvoDF(self, key_val):
        match_idx_df = self.text_matches_new[key_val][1]  # This contains 'row_idx' and 'matchfreq'
        df_main = self.getDataframe()  # Main convo dataframe
        # Find unique 'row_idx' values where 'matchfreq' is nonzero
        matched_row_idxs = match_idx_df.loc[match_idx_df['match_freq'] != 0, 'row_idx'].unique()
        # Filter self.getDataframe() where 'row_idx' is in matched_row_idxs
        matched_convos = df_main[df_main.index.isin(matched_row_idxs)]
        return matched_convos

    def checkAI(self, row_idx):
        df = self.df
        AI_seller_str = "Your sudden demand for a refund is unwarranted. Our product description is crystal clear, and we stand by our policy. Your behavior is disappointing, and your negative review is unfounded."
        AI_Buyer_str = "Your response is utterly unacceptable. I bought the jersey for my nephew, a Kobe Bryant fan, based on your explicit representation. Your deceptive behavior is disgraceful."
        # Check if AI_seller_str is in any row of formattedChat
        # print(df["formattedChat"][row_idx])
        AI_seller_match = AI_seller_str.lower() in df["formattedChat"][row_idx].lower()
        AI_buyer_match = AI_Buyer_str.lower() in df["formattedChat"][row_idx].lower()

        if AI_seller_match:
            return "Seller"
        elif AI_buyer_match:
            return "Buyer"
        else:
            return None
    
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
                        'speaker_id': spk+ '_'+str(row_idx),
                        'is_AI':    None
                            })
            return
        pattern = re.compile(r'^\s*(\d+|nan)?\s*(Buyer|Seller):\s*(.*)$', re.IGNORECASE)
        lines = str(row_entry).split('\n')
        # Determine if the first line indicates AI involvement
        first_line = lines[0].strip() if lines else ""
        # print("first line is:", first_line , "\n")
        ai_speaker = self.checkAI(row_idx)  # "seller", "buyer", or None
        # print("first speaker AI is:", ai_speaker , "\n")
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
                 # Determine if the current speaker is AI based on ai_speaker
                is_AI = (str(ai_speaker).lower() == "seller" and str(speaker).lower() == "seller") or \
                (str(ai_speaker).lower() == "buyer" and str(speaker).lower() == "buyer")
                # print("is AI is:", is_AI , "\n")
                structured_dialog.append({
                    'timestamp': timestamp,
                    'speaker': speaker,
                    'message': message.strip(),
                    'value': None,
                    'uttidx': line_count,
                    'speaker_id': speaker + '_' + str(row_idx) if speaker is not None else None,
                    'is_AI': is_AI

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
                        'speaker_id': spk + '_' + str(row_idx) if spk is not None else None,
                        'is_AI': ai_speaker
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
    def filterMatches(self, col_name, value_to_check, subset_to_exclude = None, case_in= None, case_ex= None):
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
        
        df = self.utterancesDF
        # Ensure the column is string type, replacing NaN values with empty strings
        df[col_name] = df[col_name].fillna("").astype(str)

        # Create Boolean masks for each match type
        exact_match = df[col_name].str.contains(value_to_check, na=False)
        lower_match = df[col_name].str.contains(r'\b' + re.escape(value_to_check.lower()) + r'\b', na=False) & ~exact_match
        case_insensitive_match = df[col_name].str.contains(r'\b' + re.escape(value_to_check) + r'\b', case=False, na=False) & ~exact_match & ~lower_match
        df.loc[exact_match, "Case Match Type"] = "Exact"
        df.loc[lower_match, "Case Match Type"] = "Lower"
        df.loc[case_insensitive_match, "Case Match Type"] = "Case Insensitive"
        df.loc[exact_match | lower_match | case_insensitive_match, "match_idx"] = True 
  
        if subset_to_exclude:
            included_rows = self.filterRows(col_name, value_to_check, subset_to_exclude, case_in, case_ex)
            df["match_idx"] = df.index.isin(included_rows.index)  # This sets match_idx True for rows in df_include, False elsewhere.
            mask = ~df.index.isin(included_rows.index)
            df.loc[mask, "Case Match Type"] = None
    
        # Filter matched utterances (keep same length as self.df
        matched_df_utt = df[["row_idx", "match_idx", "Case Match Type", "convo_len"]]
       
        #changes filtered df to exclude the subset for the matched row statistics 
        if subset_to_exclude:
            utt_stats = included_rows[included_rows["match_idx"] == True][["row_idx", "uttidx", "Case Match Type", "convo_len", "is_AI"]]
        else:
            utt_stats = df[df["match_idx"] == True][["row_idx", "uttidx", "Case Match Type", "convo_len", "is_AI"]]
        # Copute match frequency (count matches per row_idx group)
        match_freq = df.groupby("row_idx")['Case Match Type'].apply(lambda x: x.isin(["Exact", "Lower", "Case Insensitive"]).sum()).reset_index(name="match_freq")

        # Compute conversation length per row_idx
        convolen = df.groupby("row_idx").size().reset_index(name='convo_len')
        # Merge match_freq and convolen on row_idx to create the summary DataFrame
        convo_stats = pd.merge(match_freq, convolen, on="row_idx", how="left")   
        convo_stats.head() 
        self.text_matches_new[value_to_check] = [matched_df_utt, convo_stats, utt_stats]
        self.resetUtDF()
        self.normalizedRelativePos(value_to_check)

    def resetUtDF(self):
        df = self.utterancesDF
        df['Case Match Type'] = np.nan
        df['Case Match Type'] = df['Case Match Type'].astype('object')
        df['match_idx'] = False
 
    def getConvoMatchesByCase(self, value_key):
        df_utt = self.utterancesDF
        df_utt['Case Match Type'] = self.text_matches_new[value_key][2]['Case Match Type']
        # df_utt.groupby("row_idx")['Case Match Type'].apply(lambda x: x.isin(["Exact", "Lower", "Case Insensitive"]).sum())
        group = df_utt.groupby("row_idx")["Case Match Type"].value_counts().unstack(fill_value=0)
        df_utt['Case Match Type'] = np.nan
        df_utt['Case Match Type'] = df_utt['Case Match Type'].astype('object')
        print(f"\n'{value_key}` Total Number of Case Match Types Across Utterances")
        return group.sum().to_frame(name="Total Count").reset_index()

    def groupbyMatchUttStat(self, value_key, group_by, stat_cols, agg_list):
        # df_utt = self.getMatchedUtterancesDF(value_key).copy()
        # df_utt.loc[:,stat_col] = self.text_matches_new[value_key][2][stat_cols] # Assuming this exists
        # stat = df_utt.groupby(group_by)[stat_cols].agg(agg_list)
        # print(f"Key Value: {value_key}, Grouped by: {group_by}, Aggregated column: {stat_cosl}, Aggregations: {agg_list}")
        """
        This function groups by the given columns and applies aggregation functions to the provided statistic columns.
        """
        df_utt = self.getMatchedUtterancesDF(value_key)
        df_utt.loc[:,stat_cols] = self.text_matches_new[value_key][2][stat_cols]
         # Build the aggregation dictionary using the original column names
        agg_dict = {col: agg_list[i] for i, col in enumerate(stat_cols)}
        stat = df_utt.groupby(group_by).agg(agg_dict)
        rename_dict = {col: f"{agg_list[i]}_{col}" for i, col in enumerate(stat_cols)}
        stat.rename(columns=rename_dict, inplace=True)
        print(f"Key Value: {value_key}, Grouped by: {group_by}, Aggregated columns: {stat_cols}, Aggregations: {agg_dict}")
        return stat
    
    def groupbyMatchConvoStat(self, value_key, group_by, stat_cols, agg_list):
        df_convo = self.getMatchedConvoDF(value_key)
        df_convo.loc[:,stat_cols] = self.text_matches_new[value_key][1][stat_cols]  # Assuming this exists
        # Build the aggregation dictionary using the original column names
        agg_dict = {col: agg_list[i] for i, col in enumerate(stat_cols)}
        stat = df_convo.groupby(group_by).agg(agg_dict)
        rename_dict = {col: f"{agg_list[i]}_{col}" for i, col in enumerate(stat_cols)}
        stat.rename(columns=rename_dict, inplace=True)
        print(f"Key Value: {value_key}, Grouped by: {group_by}, Aggregated column: {stat_cols}, Aggregations: {agg_list}")
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
    
        filtered_df = self.utterancesDF  # Start with the full DataFrame
    
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
                if col_name.lower().startswith("b"):
                    spk = "Buyer"
                elif col_name.lower().startswith("s"):
                    spk = "Seller"
                else:
                    spk = None
                return spk


    ''' All statistics for words'''
    # def speakerPhrase(self):
    def getTTR(self, key_value, matched):
        df = self.text_matches_new[key_value][1]
        if matched:
            df = df[df[['match_freq'] != 0]]
        convos_with_phrase = df['match_freq'].sum()
      
        return pd.to_numeric(convos_with_phrase/ (df.shape[0]))
        
    def getDispersion(self, key_val, matched):
        #f_i  = frequency of phrase in conversation 
        #f_bar mean frequency across all conversations.
        df = self.text_matches_new[key_val][1]
        if matched:
            df = df[df['match_freq'] != 0]

        f_mean  = df['match_freq'].mean()
        squared_dev = ((df['match_freq'] - f_mean) ** 2).sum()
        print("f_mean is:", f_mean)
        print(" squared dev is:", squared_dev)
        dispersion_index = ((df['match_freq'] - f_mean) ** 2).sum() / f_mean
        return dispersion_index

    def getEntropy(self, key_val, matched):
        df = self.text_matches_new[key_val][1]
   
        if matched:
            df = df[df[['match_freq'] != 0]]
        phrase_counts = df['match_freq']
        phrase_probs = phrase_counts / phrase_counts.sum()
        phrase_entropy = entropy(phrase_probs, base=2)
        return phrase_entropy

    #measures whether a phrase or feature is uniformly distributed
    #across conversations or if it is concentrated in a few conversations
    def getStandardizedDispersion(self, key_val, matched):
        df = self.text_matches_new[key_val][1]     
        if matched:
            df = df[df['match_freq'] != 0]
        f_mean  = df['match_freq'].mean()
        std_dev = df['match_freq'].std()
        num_conversations = len(df['match_freq'])
        # print(num_conversations)
        # print(f_mean)
        juilland_d = 1 - (std_dev / (f_mean * np.sqrt(num_conversations)))
        return juilland_d
    #always matched
    def normalizedRelativePos(self, key_value):
        #save this column to getMatchedUtterancesDF
        df = self.getMatchedUtterancesDF(key_value)
        df_2 = self.text_matches_new[key_value][2] #stats_df for matched cases
        df_2['relative_pos'] = df_2["uttidx"] / df_2["convo_len"].replace(1, float('nan'))
        df_2['relative_pos'] = pd.to_numeric(df_2["uttidx"] / (df_2["convo_len"] - 1))
        # self.text_matches_new[key_value][2] = df_2
    
    def getUttStat(self, key_value, col_name):
        return self.text_matches_new[key_value][2][col_name].to_frame()

    def addUttStat(self,key_value, col_name, col):
        self.text_matches_new[key_value][2][col_name] = col

    def addConvoStat(self,key_value, col_name, col):
        self.text_matches_new[key_value][1][col_name] = col

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

