import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

np.random.seed(101)
tf.set_random_seed(101)
#load random traning data
def loadData():
	x = np.linspace(0,50,50)
	y = np.linspace(0,50,50)
	x += np.random.uniform(-4,4,50)
	y += np.random.uniform(-4,4,50)
	return x, y 

def getmodel():
	p = tf.placeholder("float")
	q = tf.placeholder("float")
	w = tf.Variable(np.random.randn(), "w")
	b = tf.Variable(np.random.randn(), "b")
	learning_rate = 0.01
	training_epoch = 1000
	n = 50
	y_pred = tf.add(tf.multiply(p, w), b)
	cost = tf.reduce_sum(tf.pow((q - y_pred), 2))/(2*n)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	
	return locals()

def main():
    X, Y = loadData()
    plt.scatter(X, Y, label="Original Smaple data")
    plt.title("Regression analysis")
    plt.show()

    #model = getmodel()
    p = tf.placeholder("float")
    q = tf.placeholder("float")
    w = tf.Variable(np.random.randn(), "w")
    b = tf.Variable(np.random.randn(), "b")
    learning_rate = 0.01
    training_epoch = 1000
    n = 50
    y_pred = tf.add(tf.multiply(p, w), b)
    cost = tf.reduce_sum(tf.pow((q - y_pred), 2)) / (2*n)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    training_epoch = 1000
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(0,training_epoch):
            for(i, j) in zip(X, Y):
                sess.run(optimizer, feed_dict={p : i, q : j})

            if((epoch+1)%50 == 0):
                c = sess.run(cost, feed_dict = {p: X, q: Y})
                print("Epoch: ", (epoch+1), "Cost = ", c, ", W = ", sess.run(w), ", b = ", sess.run(b)) 

        training_cost = sess.run(cost, feed_dict ={p: X, q: Y})
        weight = sess.run(w) 
        bias = sess.run(b)

    predictions = weight * X + bias 
    print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')

    # Plotting the Results 
    plt.plot(X, Y, 'ro', label ='Original data') 
    plt.plot(X, predictions, label ='Fitted line') 
    plt.title('Linear Regression Result') 
    plt.legend() 
    plt.show()


if __name__ == "__main__":
	main()