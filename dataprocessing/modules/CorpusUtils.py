from convokit import download, Corpus
import pandas as pd
import convokit

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
    def buildCorpus():

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
  