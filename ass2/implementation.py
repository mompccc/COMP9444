import tensorflow as tf
import re

BATCH_SIZE = 64
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider/max time
EMBEDDING_SIZE = 50  # Dimensions for each word vector/steps
HIDDEN_SIZE = 50

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than', 'br', 'ie', 'but', 'e', 'g'})

def preprocess(review):
    review = review.lower()
    review = re.sub(r"\'s", "", review)
    review = re.sub(r"[^a-z\-\s]", " ", review)
    reviews = review.split()
    processed_review = []
    for word in reviews:
        if word not in stop_words:
            temp = re.findall('[a-z]', word)
            if temp:
                if re.match('^-', word):
                    word = word[1:]
                if re.match('-$', word):
                    word = word[:-1]
                processed_review.append(word)
    processed_review.reverse()
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    return processed_review


def define_graph():
    input_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name='input_data')
    labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 2], name='labels')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    w = {
        'in': tf.Variable(tf.random_normal([MAX_WORDS_IN_REVIEW, HIDDEN_SIZE])),
        'out': tf.Variable(tf.random_normal([HIDDEN_SIZE, 2]))
    }
    b = {
        'in': tf.Variable(tf.constant(0.1, shape=[HIDDEN_SIZE, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[2, ]))
    }

    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE/2, forget_bias=1.0)
    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, forget_bias=1.0)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell_1, lstm_cell_2])
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8, input_keep_prob=0.8)
    init_state = cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)

    input_data_new = tf.reshape(input_data, [-1, MAX_WORDS_IN_REVIEW])
    input_in = tf.matmul(input_data_new, w['in']) + b['in']
    #input_in = tf.nn.dropout(input_in, 0.8)
    input_in = tf.reshape(input_in, [-1, EMBEDDING_SIZE, HIDDEN_SIZE])

    outputs, states = tf.nn.dynamic_rnn(cell, input_in, initial_state=init_state, time_major=False)
    last_output = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    preds = tf.matmul(last_output[-1], w['out']) + b['out']
    #preds = tf.nn.dropout(preds, 0.8)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=preds), name='loss')
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss=loss)

    prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"), name='accuracy')

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss

