## Basic Directory Structure
Corpus -> Conversation -> utterance -> speaker
corpus_directory
      |-- utterances.jsonl
      |-- speakers.json
      |-- conversations.json
      |-- corpus.json
      |-- index.json
metadata and vectors for task-specific properties
<mark class="hltr-purple">UtteranceNode:</mark> Wrapper class around Utterances to facilitate tree traversal operations
<mark class="hltr-purple">.meta</mark>: can store any object as metadata value, the format is a dictionary
- Immutable 
<mark class="hltr-purple">{corpus-Object}-index:</mark> Don't need to load entire dataset, just get access to metadata.
 **Pickling** Making data into binary file, platform dependent.
## Basic Model 
Classes:
- Speaker
- Utterance 
- Conversation
- Corpus
- ConvoKitMatrix
- UtteranceNode
### Utterances
Each utterance is stored on its own line and represented as a json object, with six mandatory fields:
- id: index of the utterance
- speaker: the speaker who authored the utterance
- conversation_id: id of the first utterance in the conversation this utterance belongs to
- reply_to: index of the utterance to which this utterance replies to (None if the utterance is not a reply)
- timestamp: time of the utterance
- text: textual content of the utterance
### Conversations
- `check_integrity`(_verbose: bool = True_) → bool
	- do the constituent utterances form a complete reply-to chain?
	- Have the option of making a "reply-to" chain function to annotate the conversation flow

### ConvokitMatrix
Contains vectors for a collection of corpus component objects, which is stored in the Corpus object, as opposed to individual utterances / speakers / conversations. This allows it to be readily used as a matrix as needed. At the same time, with some nifty engineering, the vector for any corpus component object can be accessed directly from the object itself. The ConvoKitMatrix object also stores mappings from rows to object ids and mappings from columns to column names, allowing for easy interpretation of the meaning of these matrices.

### Transformers:
- Transformers come under three categories: preprocessing, feature extraction, and analysis.
<mark class="hltr-purple">Transformer Class:</mark> Modifies the corpus with some changed metadata
- Abstract class that contains fit and transform methods
- Abstract classes can have shared functionality
- <mark class="hltr-yellow">parser.transform(corpus)</mark>
- Can chain together learn transformer with ConvoKit, also use <mark class="hltr-purple">Pipeline </mark>
- Can create custom **transformer class** 
## Storage 
- during runtime, stored as python object
- potential difficult bc need a lot of RAM with large datasets
- MongoDB Option:
	- lazy loading
	- disk operations for reading-> slower
# Resources
ConvoKit
 - https://convokit.cornell.edu/
How to construct a dataset object
 - https://github.com/CornellNLP/ConvoKit/blob/master/examples/converting_movie_corpus.ipynb
ConvoKit Datasets
 - https://zissou.infosci.cornell.edu/convokit/datasets/
ConvaKit Classes
	 - https://convokit.cornell.edu/documentation/model.html
Vector Demo
 - https://github.com/CornellNLP/ConvoKit/blob/master/examples/vectors/vector_demo.ipynb
Model Documentation:
 - https://convokit.cornell.edu/documentation/model.html
Transformer Subclasses:
 - https://convokit.cornell.edu/documentation/transformers.html
Conversation Forecasting Dataset:
 - https://convokit.cornell.edu/documentation/awry.html
 - https://convokit.cornell.edu/documentation/awry_cmv.html (change my view subreddit)
