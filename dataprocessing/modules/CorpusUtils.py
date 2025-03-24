import re
import pandas as pd
from convokit import Corpus, Speaker, Utterance
class CorpusUtils:
    """
    Utility class for processing and building dataframes for conversations, speakers, and utterances.
    """
    def __init__(self):
        self.utterance_headers= ["id", "speaker", "conversation_id", "reply_to", "timestamp", "text"]
        # TODO: add the rankings for each outcome later
        self.speaker_metadata = ["b_country", "s_country", "is_AI"]

        self.conversation_headers = ["id", "name", "timestamp", "num_utterances"]
        self.conversation_metadata = ["num_turns", "dispute"]
    # intention: make dataframe object from a pre-processed dataframe with correct primary info and metadata info needed

    def setUtteranceHeaders(self, headers):
        self.utterance_headers = headers

    def setSpeakerHeaders(self, headers):
        self.speaker_metadata = headers
    
    def setConversationHeaders(self, headers):
        self.conversation_headers = headers

    def setConversationMetadata(self, metadata):
        self.conversation_metadata = metadata   
    
    def setSpeakerMetadata(self, metadata):
        self.speaker_metadata = metadata


    @staticmethod
    def buildspeakerID(df):

    @staticmethod
    def buildconversationID(df):

    @staticmethod
    def buildUtteranceDataFrame(df):
        """
        Creates utterance dataframe with all primary attributes for an utterance from a formatted DataFrame.
        Attributes:
        -----------
        id : str or int
            Unique identifier for the utterance.
        speaker : str
            The speaker who made the utterance.
        conversation_id : str or int
            The ID of the conversation this utterance belongs to.
        reply_to : str or int or None
            The ID of the utterance this one is replying to (None if it's the first message).
        timestamp : str or int or datetime
            The time when the utterance was made.
        text : str
            The content of the utterance.
        
        Parameters:
        -----------
        df : pandas.DataFrame
        Returns:
        --------
        df : pandas.DataFrame
        """
        
        return {
            "id": row["id"],
            "speaker": row["speaker"],
            "conversation_id": row["conversation_id"],
            "reply_to": row.get("reply_to", None),
            "timestamp": row["timestamp"],
            "text": row["text"]
        }

    @staticmethod
    def buildSpeakerDataframe(df):
    """
    Creates speaker dataframe with all primary attributes for a speaker from a formatted DataFrame.
    Attributes:
    -----------
    id : str or int
        Unique identifier for the speaker.
    country : str
        The country of the speaker.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    Returns:
    --------
    df : pandas.DataFrame
    """
    
    return {
        "id": row["id"],
        "country": row["country"]
    }

# def toConversations():
  
# def toCorpus():
  
# def toIndex():
  

def parse_conversations_from_dataframe(
    df: pd.DataFrame,
    chat_col: str = 'formattedChat'
):
    """
    Parses each row in df[chat_col] as a separate conversation in ConvoKit format,
    ensuring that each Buyer or Seller in a row is *unique* to that row.
    """

    all_utterances = []

    for row_idx, chat_data in df[chat_col].items():
        # Skip if empty
        if pd.isna(chat_data) or not isinstance(chat_data, str):
            continue

        # We'll treat each row as ONE conversation
        conversation_id = f"conv_{row_idx}"

        # Split the entire conversation text by newline
        lines = chat_data.strip().split('\n')

        # A local speaker cache for just this row:
        #  - "Buyer" -> Speaker("Buyer_rowX")
        #  - "Seller" -> Speaker("Seller_rowX")
        row_speakers = {}

        prev_utt_id = None  # for linking replies

        for line_idx, line in enumerate(lines):
            # Pattern: "1699388150 Buyer: message..." or "nan Seller: message..."
            pattern = r"^(\S+)\s+(Buyer|Seller):\s(.*)"
            match = re.match(pattern, line.strip())
            if not match:
                # Could log or ignore lines that don't match
                continue

            timestamp_str, speaker_name, text = match.groups()

            # Convert timestamp, if valid:
            if timestamp_str.lower() != 'nan':
                try:
                    timestamp = int(timestamp_str)
                except ValueError:
                    timestamp = None
            else:
                timestamp = None

            # If this speaker_name hasn't appeared yet in this row,
            # create a new unique Speaker id for them
            if speaker_name not in row_speakers:
                unique_speaker_id = f"{speaker_name}_row{row_idx}"
                row_speakers[speaker_name] = Speaker(id=unique_speaker_id)

            speaker_obj = row_speakers[speaker_name]

            # Unique utterance ID
            utt_id = f"utt_{row_idx}_{line_idx}"

            # Build the Utterance
            new_utt = Utterance(
                id=utt_id,
                speaker=speaker_obj,
                conversation_id=conversation_id,
                reply_to=prev_utt_id,
                timestamp=timestamp,
                text=text
            )
            all_utterances.append(new_utt)

            # Link to the next line
            prev_utt_id = utt_id

    # Build the Corpus
    corpus = Corpus(utterances=all_utterances)
    return corpus


def convert_dataframe_to_convokit(df: pd.DataFrame, output_dir: str):
    """
    High-level function that parses the chat data from a dataframe 
    and saves it as a ConvoKit corpus on disk.
    """
    corpus = parse_conversations_from_dataframe(df, chat_col='formattedChat')

    # Optionally add some corpus-level metadata
    corpus.meta['source'] = "My Buyer-Seller Chat"
    corpus.meta['description'] = "An example corpus of buyer-seller chat logs where each row's Buyer/Seller is unique."

    corpus.dump(output_dir)
    print(f"Corpus saved to '{output_dir}'.")