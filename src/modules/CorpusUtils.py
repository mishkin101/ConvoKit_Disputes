import re
import pandas as pd
from convokit import Corpus, Speaker, Utterance, Conversation

utterance_headers = ["id", "speaker", "conversation_id", "reply_to", "timestamp", "text"]
utterance_metadata = None
speaker_headers = ['id']
speaker_metadata = None#["b_country", "s_country", "is_AI"]
conversation_headers = ['id']  #["id", "name", "timestamp", "num_utterances"]
conversation_metadata = None #["num_turns", "dispute"]


# intention: make dataframe object from a pre-processed dataframe with correct primary info and metadata info needed 
#ID, timestamp, text, speaker (a string ID), reply_to (a string ID), conversation_id (a string ID).

def setUtteranceHeaders(headers):
   global utterance_headers
   utterance_headers = headers

def setSpeakerHeaders(headers):
    global speaker_metadata
    speaker_metadata = headers

def setConversationHeaders(headers):
    global conversation_headers
    conversation_headers = headers

def setConversationMetadata(metadata):
    global conversation_metadata
    conversation_metadata = metadata   

def setSpeakerMetadata(metadata):
    global speaker_metadata 
    speaker_metadata = metadata

def convertHeaders(df, corpus_type):
    if corpus_type.lower() == 'utterance':
        rename_map = {
        'speaker_id': 'speaker',
        'timestamp': 'timestamp',
        'message': 'text'
        }
        df = df.rename(columns=rename_map)
        if utterance_metadata is not None:
            all_cols = utterance_headers + utterance_metadata
            df = prepend_meta(df, utterance_metadata)
            df = df[all_cols]
        else:
            df = df[utterance_headers]
        return df
    
    if corpus_type.lower() == 'conversation':
        df['id'] = df.index
        if conversation_metadata is not None:
            all_cols =conversation_headers +conversation_metadata
            df =prepend_meta(df, conversation_metadata)
            df = df[all_cols]
        else:
            df = df[conversation_headers]
        return df

    if corpus_type.lower() == 'speaker':
        rename_map = {
        'speaker_id': 'id',
        }
        df = df.rename(columns=rename_map)
        if speaker_metadata is not None:
            all_cols =speaker_headers +speaker_metadata
            df =prepend_meta(df,speaker_metadata)
            df = df[all_cols]
        else:
            df = df[speaker_headers]
        return df

def setReplyTo(df):
    df['reply_to'] = df['id'].shift(1)  # Set reply_to to the previous 'id'
    df.loc[df['uttidx'] == '0', 'reply_to'] = None
    display()
    return df['reply_to']

def prepend_meta(df, meta_list):
    for col in df.columns:
        if col in meta_list:
            df.rename(columns={col: 'meta.' + col}, inplace=True)
    return df
    
def buildUtteranceDF(df):
    df = df.copy()
    df.drop('speaker', axis=1, inplace=True)
    df[['row_idx', 'uttidx']]= df[['row_idx', 'uttidx']].astype(str)
    df['id'] = df.apply(lambda row: f"utt{row['uttidx']}_con{row['row_idx']}", axis=1)
    df['reply_to'] = setReplyTo(df)
    # df.drop('uttidx', axis=1, inplace=True)
    df['conversation_id'] = df.groupby('row_idx')['id'].transform('first')
    df = convertHeaders(df,'utterance')
    return df

def buildSpeakerDF(df):
    df = df.copy()
    df =convertHeaders(df,'speaker')
    return df[['id']].drop_duplicates().reset_index(drop=True)
    #ifspeaker_metadata:

def buildConvoDF(df):
    df = df.copy()
    df =convertHeaders(df,'conversation')
    return df[['id']].drop_duplicates().reset_index(drop=True)

def is_valid_timestamp(val):
    try:
        return val is not None and str(val).lower() != 'nan' and pd.notnull(val)
    except:
        return False

def clean_corpus_timestamps(data, utts, speakers, convos):
    # Filter utterances with valid timestamps
    utts = utts[utts["timestamp"].apply(is_valid_timestamp)].copy()
    utts["timestamp"] = utts["timestamp"].astype(int)

    # Match speakers and conversations to cleaned utterances
    used_speakers = set(utts["speaker"].unique())
    used_convos = set(utts["conversation_id"].unique())

    speakers = speakers[speakers["id"].isin(used_speakers)].reset_index(drop=True)
    convos = convos[convos["id"].isin(used_convos)].reset_index(drop=True)

    return utts, speakers, convos


def corpusBuilder(data):
    utts =buildUtteranceDF(data.getUtterancesDF())
    speakers =buildSpeakerDF(data.getUtterancesDF())
    convos =buildConvoDF(data.getDataframe())
    utts, speakers, convos = clean_corpus_timestamps(data, utts, speakers, convos)
    corpus_ob = Corpus.from_pandas(utterances_df=utts, speakers_df=speakers, conversations_df=convos)
    return corpus_ob

'''return: Speaker DataFrame'''
def buildspeakerParams(self, df):
    speakers_dict = {}
    # We'll store all utterances and conversations by speaker base id
    grouped = df.groupby("speaker_id")
    for full_speaker_id, group in grouped:
        utts = {}
        convos = {}
        for _, row in group.iterrows():
            utt_id = f"{row['uttidx']}"  # unique across dataset
            conv_id = f"{row['row_idx']}"
            utterance = Utterance(
                id=utt_id,
                speaker=full_speaker_id,
                conversation_id=conv_id,
                text=row["message"]
            )
            utts[utt_id] = utterance
            # Only create conversation if not already added
            if conv_id not in convos:
                convos[conv_id] = Conversation(id=conv_id, utterances=[utterance])
            else:
                convos[conv_id].add_utterance(utterance)

        if full_speaker_id not in speakers_dict:
            speakers_dict[full_speaker_id] = {
                "id": full_speaker_id,
                "utts": utts,
                "convos": convos
            }
        else:
            speakers_dict[full_speaker_id]["utts"].update(utts)
            speakers_dict[full_speaker_id]["convos"].update(convos)

    return list(speakers_dict.values())

    speaker_list=[]

if __name__ == "__main__":
    print("This file")