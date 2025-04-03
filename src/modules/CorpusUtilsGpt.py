import re
import pandas as pd
from convokit import Corpus, Speaker, Utterance
import spacy

def parse_conversations_from_dataframe(
    df: pd.DataFrame,
    chat_col: str = 'formattedChat'):
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
    
def parse_chat(chat_text):
    # Convert to string to avoid errors (NaN -> 'nan' -> we'll treat that like empty)
    if pd.isnull(chat_text):
        chat_text = ""  # or "No chat available"
    
    pattern = re.compile(r'^(\d+)\s+(Buyer|Seller):\s+(.*)$')
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
        else:
            if structured_dialog and not line.startswith("Submitted agreement:"):
                structured_dialog[-1]['message'] += " " + line
            else:
                structured_dialog.append({
                    'timestamp': None,
                    'speaker': None,
                    'message': line
                })
    return structured_dialog

def extract_outcome_info(dialog_list):
    """
    Looks for a line that starts with "Submitted agreement:" 
    and tries to parse key outcomes (refund type, apologies, review retraction).
    Returns a dict with extracted info.
    """
    outcome = {
        'agreement_line': None,
        'buyer_refund_type': None,  # e.g. "partial", "full", etc.
        'buyer_retracted_review': False,
        'seller_retracted_review': False,
        'buyer_apologized': False,
        'seller_apologized': False
    }
    
    # 1. Look for the "Submitted agreement" line
    for entry in dialog_list:
        line = entry['message']
        if line.startswith("Submitted agreement:"):
            outcome['agreement_line'] = line
            # Try to parse some known elements from the string
            # Example:
            # "Submitted agreement: Buyer gets partial refund, buyer retracted their review, seller retracted their review, buyer did apologize, and seller did apologize."
            if "partial refund" in line.lower():
                outcome['buyer_refund_type'] = "partial"
            elif "full refund" in line.lower():
                outcome['buyer_refund_type'] = "full"
            
            if "buyer retracted their review" in line.lower():
                outcome['buyer_retracted_review'] = True
            if "seller retracted their review" in line.lower():
                outcome['seller_retracted_review'] = True
            if "buyer did apologize" in line.lower():
                outcome['buyer_apologized'] = True
            if "seller did apologize" in line.lower():
                outcome['seller_apologized'] = True
            
            break  # We found a "Submitted agreement" line, so exit
    
    # 2. If we want to detect apologies outside the agreement line
    #    or detect them in the conversation at large:
    buyer_apology_words = {"apology", "apologize", "sorry"}
    seller_apology_words = {"apology", "apologize", "sorry"}

    # We can do a quick pass over all lines and see if buyer or seller
    # uses an apology phrase. (If you prefer to rely only on agreement line, skip this.)
    for entry in dialog_list:
        if entry['speaker'] == 'Buyer':
            # check if any apology word is in the message
            if any(word in entry['message'].lower() for word in buyer_apology_words):
                outcome['buyer_apologized'] = True
        elif entry['speaker'] == 'Seller':
            if any(word in entry['message'].lower() for word in seller_apology_words):
                outcome['seller_apologized'] = True

    return outcome

def lemmatize_text(text):
    nlp = spacy.load("en_core_web_sm")
    # If the text is NaN, return as-is
    if pd.isna(text):
        return text
    
    doc = nlp(text)
    # Filter out stopwords, punctuation, and spaces, then lemmatize
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens)


if __name__ == "__main__":
    ## bunch of testing code from the original notebook below
    
    
    # Test the functions
    # df['lemmatized_chat'] = df['formattedChat'].apply(lemmatize_text)

    # Quick look at the new column
    # print(df[['formattedChat', 'lemmatized_chat']].head())
    df = pd.read_csv('../data/alldyads.csv', header=0)

    # Convert all NaN to empty strings right in the DataFrame
    df['formattedChat'] = df['formattedChat'].fillna("")

    parsed_dialogs = []
    for _, row in df.iterrows():
        chat_text = row['formattedChat']
        dialog_list = parse_chat(chat_text)
        parsed_dialogs.append(dialog_list)

    df['parsed_dialog'] = parsed_dialogs
    df.head()

    ## Test parsing & outcome extraction
    chat = pd.DataFrame(df['formattedChat'])

    # We'll parse the entire DataFrame
    parsed_dialogs = []
    outcomes = []

    for i, row in chat.iterrows():
        chat_text = row['formattedChat']
        dialog_list = parse_chat(chat_text)
        parsed_dialogs.append(dialog_list)
        outcome_info = extract_outcome_info(dialog_list)
        outcomes.append(outcome_info)

    # Convert parsed_dialogs into a column if you like, 
    # or store it as a separate structure. 
    # outcomes is a list of dicts with extracted info for each row in df.
    chat['parsed_dialog'] = parsed_dialogs
    chat_outcomes = pd.DataFrame(outcomes)
    chat_final = pd.concat([chat, chat_outcomes], axis=1)

    # print("Parsed Results:")
    # print(chat_final[['formattedChat', 'parsed_dialog',
    #                 'buyer_refund_type', 'buyer_retracted_review',
    chat_final
    #                 'seller_retracted_review', 'buyer_apologized', 'seller_apologized']])