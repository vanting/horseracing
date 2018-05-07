from numpy import exp, array, random, dot
import numpy as np
import pandas as pd
import xlrd
from sklearn import preprocessing
import mysql.connector
from decimal import *

from mysql.connector import errorcode


def swap(ranking, y):
    temp = ranking[y][0]
    temp1 = ranking[y][1]
    temp2 = ranking[y][2]
    temp3 = ranking[y][3]
    temp4 = ranking[y][4]
    temp5 = ranking[y][5]
    temp6 = ranking[y][6]

    ranking[y][0] = ranking[y + 1][0]
    ranking[y][1] = ranking[y + 1][1]
    ranking[y][2] = ranking[y + 1][2]
    ranking[y][3] = ranking[y + 1][3]
    ranking[y][4] = ranking[y + 1][4]
    ranking[y][5] = ranking[y + 1][5]
    ranking[y][6] = ranking[y + 1][6]

    ranking[y + 1][0] = temp
    ranking[y + 1][1] = temp1
    ranking[y + 1][2] = temp2
    ranking[y + 1][3] = temp3
    ranking[y + 1][4] = temp4
    ranking[y + 1][5] = temp5
    ranking[y + 1][6] = temp6






# estimated function
def A_expect(Avalue,ranking, n):
    value = (n * (n - 1) / 2)
    total=0
    for x in range(n):
        total=total +(1/(1+(10**((ranking[x][0]-Avalue)/400))))

    return (total/value)


# elo rating system
def rating(value,rank, result,  n,type):
    if (type ==1):
        score = value + 7 * (Svalue(result, n,type) - A_expect(value, rank, n))
    else:
        score = value + 2 * (Svalue(result,n,type) - A_expect(value,rank, n))
    return score

# score function
def Svalue(result,n,type):
    if (type==1):
        return (n-(result))/((n*(n-1))/2)
    else:
        return (n-(result+0.5))/(n*(n-1)/2)



class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 3* random.random((number_of_inputs_per_neuron, number_of_neurons))-1


class NeuralNetwork():
    def __init__(self, layer1, layer2,layer3):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3=layer3


    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))


    def __sigmoid_derivative(self, x):
        return x * (1 - x)


    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 , output_from_layer_3= self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer3_error = training_set_outputs - output_from_layer_3
            layer3_delta = layer3_error * self.__sigmoid_derivative(output_from_layer_3)


            layer2_error = layer3_delta.dot(self.layer3.synaptic_weights.T)
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)


            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
            layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment
            self.layer3.synaptic_weights += layer3_adjustment



    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        output_from_layer3 = self.__sigmoid(dot(output_from_layer2, self.layer3.synaptic_weights))
        return output_from_layer1, output_from_layer2,output_from_layer3

    # The neural network prints its weights
    def print_weights(self):
        print ("    Layer 1 (4 neurons, each with 3 inputs): ")
        print (self.layer1.synaptic_weights)
        print ("    Layer 2 (1 neuron, with 4 inputs):")
        print (self.layer2.synaptic_weights)



#variable for window size
starting_point =400
ending_point = 1200
verify_sample_size =40
starting=0
precal=400


# connect database
cnx = mysql.connector.connect(user='root',password='dereK412')

n=0
year1=0
year2=0
year3=0
year4=0
year5=0
year6=0
total_five= total_ten=total_fif=total_twe=total_twe_fif=total_thir=0
bet_five=bet_ten=bet_fif=bet_twe=bet_twe_fif=bet_thir=0

#variables for win odd
t1=t2=t3=t4=t5=t6=t7=t8=t9=t10=t11=t12=t13=t14=t15=t16=t17=t18=t19=t20=t21=t22=t23=t24=t25=t26=t27=t28=t29=t30=0
p1=p2=p3=p4=p5=p6=p7=p8=p9=p10=p11=p12=p13=p14=p15=p16=p17=p18=p19=p20=p21=p22=p23=p24=p25=p26=p27=p28=p29=p30=0




while(ending_point<6000):

    ranking = [[-1 for x in range(6)] for y in range(15)]
    jockey_ranking = [[-1 for x in range(6)] for y in range(15)]
    trainer_ranking = [[-1 for x in range(6)] for y in range(15)]

    size = 500000  # open a 2d array
    horse_score = [[-1 for x in range(2)] for y in range(size)]  # the second index is score of horse
    jockey_score = [[-1 for x in range(2)] for y in range(size)]
    trainer_score = [[-1 for x in range(2)] for y in range(size)]
    signal = [[-1 for x in range(2)] for y in range(size)]

    run = cnx.cursor(buffered=True)
    run.execute("SELECT * FROM fyp.combine WHERE race_id>="+str(starting)+" and race_id <="+str(ending_point))
    p=starting_point

    p=0
    arrayOfHorse=[]
    arrayOfJockey=[]
    arrayOfTrainer=[]
    num_winning=0


    # calculating elo rating system
    for row in run.fetchall():


        # assign values to corresponding factors

        if horse_score[row[22]][1] != -1:
            valueOfHorse = round(horse_score[row[22]][1], 2)
        elif horse_score[row[22]][1] == -1:
            valueOfHorse = 1500

        if jockey_score[row[38]][1] != -1:
            valueOfJockey = round(jockey_score[row[38]][1], 2)
        elif jockey_score[row[38]][1] == -1:
            valueOfJockey = 1500

        if trainer_score[row[37]][1] != -1:
            valueOfTrainer = round(trainer_score[row[37]][1], 2)
        elif trainer_score[row[37]][1] == -1:
            valueOfTrainer = 1500


            # the score for DNN
        if row[0] >= precal and row[0] < (ending_point):
            arrayOfHorse.append([valueOfHorse])
            arrayOfJockey.append([valueOfJockey])
            arrayOfTrainer.append([valueOfTrainer])


        if p == row[0]:
            ranking[row[21]-1][0] = int(valueOfHorse)
            ranking[row[21]-1][1] = row[23]
            ranking[row[21]-1][2] = row[35]
            ranking[row[21]-1][3] = row[36]
            ranking[row[21]-1][4]=row[22]

            jockey_ranking[row[21]-1][0] = int(valueOfJockey)
            jockey_ranking[row[21]-1][1] = row[23]
            jockey_ranking[row[21]-1][2] = row[35]
            jockey_ranking[row[21]-1][3] = row[36]
            jockey_ranking[row[21]-1][4]= row[38]

            trainer_ranking[row[21]-1][0] = int(valueOfTrainer)
            trainer_ranking[row[21]-1][1] = row[23]
            trainer_ranking[row[21]-1][2] = row[35]
            trainer_ranking[row[21]-1][3] = row[36]
            trainer_ranking[row[21]-1][4]=row[37]

            if row[23]==1:
                num_winning=num_winning+1


            n=n+1

        else:

            for x in range(len(jockey_ranking)):
                getcontext().prec = 6
                if num_winning==1:
                    # a horse win a race
                    jockey_score[jockey_ranking[x][4]][1] = rating(valueOfJockey, jockey_ranking, jockey_ranking[x][1],  n ,1)
                    horse_score[ranking[x][4]][1] = rating(valueOfHorse, ranking,ranking[x][1],  n,2)
                    trainer_score[trainer_ranking[x][4]][1] = rating(valueOfTrainer, trainer_ranking, trainer_ranking[x][1],  n,1)
                else:

                    # two horses win a race
                    jockey_score[jockey_ranking[x][4]][1] = rating(valueOfJockey, jockey_ranking, jockey_ranking[x][1],n-1, 1)
                    horse_score[ranking[x][4]][1] = rating(valueOfHorse, ranking, ranking[x][1], n-1, 2)
                    trainer_score[trainer_ranking[x][4]][1] = rating(valueOfTrainer, trainer_ranking,trainer_ranking[x][1], n-1, 1)


            n=0
            p=p=row[0]








    input_testing = cnx.cursor(buffered=True)
    input_statement="SELECT declared_weight, actual_weight, draw, win_odds , place_odds , distance , horse_age ,horse_rating FROM fyp.combine WHERE race_id >=" + str(starting_point) + " and race_id <" + str(ending_point)

    input_testing.execute(input_statement)

    input = np.array(input_testing.fetchall())

    # combine data into a signal array
    input=np.concatenate((input,arrayOfHorse),axis=1)
    input= np.concatenate((input,arrayOfTrainer),axis=1)
    input= np.concatenate((input, arrayOfJockey),axis=1)



    output_testing = cnx.cursor(buffered=True)
    output_statement="SELECT finish_time FROM fyp.combine WHERE race_id >=" + str(starting_point) + " and race_id <" + str(ending_point)

    output_testing.execute(output_statement)
    output = np.array(output_testing.fetchall())


    y = output

    X = input
    # standardization of all data of indicators
    x_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = x_scaler.fit_transform(X)

    y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    y = y_scaler.fit_transform(y)




    if __name__ == "__main__":
        # Seed the random number generator
        random.seed(1)

        # Create layer 1 (5 neurons, each with 11 inputs)
        layer1 = NeuronLayer(5, 11)

        # Create layer 2 (2 single neuron with 5 inputs)
        layer2 = NeuronLayer(2, 5)

        layer3=NeuronLayer(1,2)

        # Combine the layers to create a neural network
        neural_network = NeuralNetwork(layer1, layer2,layer3)

        training_set_inputs = X
        training_set_outputs = y

        # Train the neural network using the training set.
        # Do it 60,000 times and make small adjustments each time.
        neural_network.train(training_set_inputs, training_set_outputs, 100000)


        arrayOfHorse=[]
        arrayOfJockey=[]
        arrayOfTrainer=[]
        run = cnx.cursor(buffered=True)
        run.execute("SELECT * FROM fyp.combine WHERE race_id>=" + str(ending_point) + " and race_id <=" + str(ending_point+verify_sample_size))

        for row in run.fetchall():

            #retrieving new score of testing data
            if horse_score[row[22]][1] != -1:
                valueOfHorse = round(horse_score[row[22]][1], 2)
            elif horse_score[row[22]][1] == -1:
                valueOfHorse = 1500
                signal[row[0]]=1

                # detecting new horses
                if row[0] > 1400 and row[0] <= 2200:
                    year1 = year1 + 1
                if row[0] > 2200 and row[0] <= 3000:
                    year2 = year2 + 1
                if row[0] > 3000 and row[0] <= 3800:
                    year3 = year3 + 1
                if row[0] > 3800 and row[0] <= 4600:
                    year4 = year4 + 1
                if row[0] > 4600 and row[0] <= 5200:
                    year5 = year5 + 1
                if row[0] > 5300 and row[0] <= 6000:
                    year6 = year6 + 1

            if jockey_score[row[38]][1] != -1:
                valueOfJockey = round(jockey_score[row[38]][1], 2)
            elif jockey_score[row[38]][1] == -1:
                valueOfJockey = 1500

            if trainer_score[row[37]][1] != -1:
                valueOfTrainer = round(trainer_score[row[37]][1], 2)
            elif trainer_score[row[37]][1] == -1:
                valueOfTrainer = 1500


                # retrieving score of testing data
            if row[0] >= ending_point and row[0] < (ending_point+verify_sample_size):
                arrayOfHorse.append([valueOfHorse])
                arrayOfJockey.append([valueOfJockey])
                arrayOfTrainer.append([valueOfTrainer])








        input_verify = cnx.cursor(buffered=True)
        input_state="SELECT declared_weight, actual_weight, draw, win_odds , place_odds , distance , horse_age ,horse_rating FROM fyp.combine WHERE race_id >=" + str(ending_point) + " and race_id <" + str(ending_point+verify_sample_size)
        input_verify.execute(input_state)
        verification_data = np.array(input_verify.fetchall())

        verification_data = np.concatenate((verification_data, arrayOfHorse), axis=1)
        verification_data = np.concatenate((verification_data, arrayOfTrainer), axis=1)
        verification_data = np.concatenate((verification_data, arrayOfJockey), axis=1)

        #standardization and prediction
        input_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        verification_data = input_scaler.fit_transform(verification_data)
        # hidden_state, output = neural_network.think(verification_data)
        hidden_state1,hidden_state2, output = neural_network.think(verification_data)



    output_verify = cnx.cursor(buffered=True)
    output_state= "SELECT race_id, result, draw, finish_time, win_odds, place_odds FROM fyp.combine WHERE race_id >=" + str(ending_point) + " and race_id <" + str(ending_point+verify_sample_size)
    output_verify.execute(output_state)

    result = np.array(output_verify.fetchall())
    #conbime the result with actual result
    people = np.concatenate((result, output), axis=1)

    i=ending_point


    num = 0
    temp_num = 0

    #sorting
    for x in range(len(people)):

        if people [x][0]!= ending_point+verify_sample_size-1:

            if people[x][0]==i :
                num=num+1

            else:
                #sorting
                for z in range(num):
                    for y in range(z):
                        if people[temp_num + y][6] > people[temp_num + y + 1][6]:
                            swap(people, temp_num + y)

                temp_num=x
                i=people[x][0]
                num = 0
        else:

            for z in range(len(people)- x):
                for y in range(z):
                    if people[ x+y][6] >people[x+y+1][6]:
                        swap(people, x+y)




    i = ending_point;
    set = 0;
    count = 0;
    temp_mean = 0;
    temp3 = 0;
    winning = 0
    money = 0
    total_win = 0
    total_money = 0
    betting_number_of_race = 0
    zero_Five = Five_ten = ten_fif = fif_twe = twe_fif = thir = 0
    money_five = money_ten = money_fif = money_twe = money_twe_fif = money_thir = 0
    w_five = w_ten = w_fif = w_twe = w_twe_fif = w_thir = 0
    race = 0


    #calculating the profit / loss
    for x in range(len(people)):
        if i != (people[x][0]):

            i = people[x][0]
            set = 0;
            temp_mean = 0;
            temp3 = 0;

        if (set < 1 )  :
            temp_mean = temp_mean + people[x][1];
            if set == 0 and signal[int (people[x][0])]!=1:
                if people[x][1] ==1:
                    total_win = total_win + 1
                    total_money = total_money + people[x][4] * 10 - 10

                else:
                    total_money = total_money - 10


                    # counting  number of betting and winning for specific win odd
                if people[x][4]>1 and people[x][4]<2:
                    t2=t2+1
                    if people[x][1]==1:
                        p2=p2+1

                elif people[x][4]>=2 and people[x][4]<3:
                    t3=t3+1
                    if people[x][1]==1:
                        p3=p3+1

                elif people[x][4] >= 3 and people[x][4] < 4:
                    t4 = t4 + 1
                    if people[x][1] ==1:
                        p4 = p4 + 1

                elif people[x][4] >= 4 and people[x][4] < 5:
                    t5 = t5 + 1
                    if people[x][1] ==1:
                        p5 = p5 + 1


                elif people[x][4] >= 5 and people[x][4] < 6:
                    t6 = t6 + 1
                    if people[x][1] ==1:
                        p6 = p6 + 1

                elif people[x][4] >= 6 and people[x][4] < 7:
                    t7 = t7 + 1
                    if people[x][1] ==1:
                        p7 = p7 + 1

                elif people[x][4] >= 7 and people[x][4] < 8:
                    t8 = t8 + 1
                    if people[x][1] ==1:
                        p8 = p8 + 1

                elif people[x][4] >= 8 and people[x][4] < 9:
                    t9 = t9 + 1
                    if people[x][1] == 1:
                        p9 = p9 + 1

                elif people[x][4] >= 9 and people[x][4] < 10:
                    t10 = t10 + 1
                    if people[x][1] ==1:
                        p10 = p10 + 1


                elif people[x][4] >= 10 and people[x][4] < 11:
                    t11 = t11 + 1
                    if people[x][1]==1:
                        p11 = p11 + 1

                elif people[x][4] >= 11 and people[x][4] < 12:
                    t12 = t12 + 1
                    if people[x][1] <=3:
                        p12 = p12 + 1

                elif people[x][4] >= 12 and people[x][4] < 13:
                    t13 = t13 + 1
                    if people[x][1] ==1:
                        p13 = p13 + 1


                elif people[x][4] >= 13 and people[x][4] < 14:
                    t14 = t14 + 1
                    if people[x][1] ==1:
                        p14 = p14 + 1

                elif people[x][4] >= 14 and people[x][4] < 15:
                    t15 = t15 + 1
                    if people[x][1] ==1:
                        p15 = p15 + 1

                elif people[x][4] >= 15 and people[x][4] < 16:
                    t16 = t16 + 1
                    if people[x][1] ==1:
                        p16 = p16 + 1

                elif people[x][4] >= 16 and people[x][4] < 17:
                    t17 = t17 + 1
                    if people[x][1] ==1:
                        p17 = p17 + 1

                elif people[x][4] >= 17 and people[x][4] < 18:
                    t18 = t18 + 1
                    if people[x][1] ==1:
                        p18 = p18 + 1

                elif people[x][4] >= 18 and people[x][4] < 19:
                    t19 = t19 + 1
                    if people[x][1] ==1:
                        p19 = p19 + 1

                elif people[x][4] >= 19 and people[x][4] < 20:
                    t20 = t20 + 1
                    if people[x][1] ==1:
                        p20 = p20 + 1

                elif people[x][4] >= 20 and people[x][4] < 21:
                    t21 = t21 + 1
                    if people[x][1] ==1:
                        p21 = p21 + 1

                elif people[x][4] >= 21 and people[x][4] < 22:
                    t22 = t22 + 1
                    if people[x][1] ==1:
                        p22 = p22 + 1

                elif people[x][4] >= 22 and people[x][4] < 23:
                    t23 = t23 + 1
                    if people[x][1] ==1:
                        p23 = p23 + 1

                elif people[x][4] >= 23 and people[x][4] < 24:
                    t24 = t24 + 1
                    if people[x][1] ==1:
                        p24 = p24 + 1

                elif people[x][4] >= 24 and people[x][4] < 25:
                    t25 = t25 + 1
                    if people[x][1] ==1:
                        p25 = p25 + 1

                elif people[x][4] >= 25 and people[x][4] < 26:
                    t26 = t26 + 1
                    if people[x][1] ==1:
                        p26 = p26 + 1

                elif people[x][4] >= 26 and people[x][4] < 27:
                    t27 = t27 + 1
                    if people[x][1] ==1:
                        p27 = p27 + 1

                elif people[x][4] >= 27 and people[x][4] < 28:
                    t28 = t28 + 1
                    if people[x][1] ==1:
                        p28 = p28 + 1

                elif people[x][4] >= 28 and people[x][4] < 29:
                    t29 = t29 + 1
                    if people[x][1] ==1:
                        p29 = p29 + 1

                elif people[x][4] >= 29 and people[x][4] < 30:
                    t30 = t30 + 1
                    if people[x][1] ==1:
                        p30 = p30 + 1




                        # calculating profit / loss
                if people[x][4] > 15 and people[x][4] <= 20:
                    bet_twe=bet_twe+1
                    betting_number_of_race = betting_number_of_race + 1
                    if people[x][1] ==1:
                        total_twe=total_twe+1
                        money = money + 10 * people[x][4] - 10
                        winning = winning + 1
                    else:
                        money = money - 10

                elif people[x][4] >= 1 and people[x][4] <= 5:
                    bet_five=bet_five+1
                    zero_Five = zero_Five + 1
                    if people[x][1] ==1:
                        total_five=total_five+1
                        money_five = money_five + 10 * people[x][4] - 10
                        w_five = w_five + 1
                    else:
                        money_five = money_five - 10

                elif people[x][4] > 5 and people[x][4] <= 10:
                    bet_ten=bet_ten+1
                    Five_ten = Five_ten + 1
                    if people[x][1] ==1:
                        total_ten=total_ten+1
                        money_ten = money_ten + 10 * people[x][4] - 10
                        w_ten = w_ten + 1
                    else:
                        money_ten = money_ten - 10


                elif people[x][4] > 10 and people[x][4] <= 15:
                    bet_fif=bet_fif+1
                    ten_fif = ten_fif + 1
                    if people[x][1] ==1:
                        total_fif=total_fif+1
                        money_fif = money_fif + 10 * people[x][4] - 10
                        w_fif = w_fif + 1
                    else:
                        money_fif = money_fif - 10

                elif people[x][4] > 20 and people[x][4] <= 25:
                    bet_twe_fif=bet_twe_fif+1
                    twe_fif = twe_fif + 1
                    if people[x][1] ==1:
                        total_twe_fif=total_twe_fif+1
                        money_twe_fif = money_twe_fif + 10 * people[x][4] - 10

                        w_twe_fif = w_twe_fif + 1
                    else:
                        money_twe_fif = money_twe_fif - 10

                elif people[x][4] > 25 and people[x][4] <= 30:
                    bet_thir=bet_thir+1
                    thir = thir + 1
                    if people[x][1] ==1:
                        total_thir=total_thir+1
                        money_thir = money_thir + 10 * people[x][4] - 10
                        w_thir = w_thir + 1
                    elif people[x][1] != 1:
                        money_thir = money_thir - 10


            set = set + 1



       # print result
    print("")
    print("period between " +str(ending_point) +" and "+str(ending_point+verify_sample_size) )
    print("")
    print("")
    print("odd 0 to 5: %3d      winning: %3d      money: %3d, " % (zero_Five, w_five, money_five))
    print("odd 5 to 10: %3d     winning: %3d      money: %3d, " % (Five_ten, w_ten, money_ten))
    print("odd 10 to 15 %3d      winning: %3d      momey: %3d" % (ten_fif, w_fif, money_fif))
    print("odd 15 to 20: %3d     winning: %3d      money: %3d, " % (betting_number_of_race, winning, money))
    print("odd 20 to 25: %3d     winning: %3d      money: %3d, " % (twe_fif, w_twe_fif, money_twe_fif))
    print("odd 25 to 30: %3d     winning: %3d      money: %3d, " % (thir, w_thir, money_thir))

    print("")
    print("total number of winning %3d     total money: %3d " % (total_win, total_money))
    print("the range is between 1 to 30: %3d" % (money_five+money_ten+money_fif+money+money_twe_fif+money_thir))
    print(" money in specific odd range between 0 to 30:%8.0f"%(money+money_twe_fif+money_thir))
    starting_point = starting_point + verify_sample_size
    ending_point = ending_point + verify_sample_size
    starting=starting+verify_sample_size
    precal=precal+verify_sample_size
print(year1,"  ",year2,"  ",year3,"  ",year4,"  ",year5,"  ",year6)


print("for 0 to 5 :", bet_five, " having ", total_five)
print("for 5 to 10 :", bet_ten, " having ", total_ten)
print("for 10 to 15 :", bet_fif, " having ", total_fif)
print("for 15 to 20 :", bet_twe, " having ", total_twe)
print("for 20 to 25 :", bet_twe_fif, " having ", total_twe_fif)
print("for 25 to 30 :", bet_thir, " having ", total_thir)

print( t2," ", t3," ", t4," ", t5," ", t6," ", t7," ", t8," ", t9," ", t10," "," ", t11," ", t12," ", t13," ", t14," ", t15," ", t16," ", t17," ", t18," ", t19," ", t20," ", t21," ", t22," ", t23," ", t24," ", t25," ", t26," ", t27," ", t28," ", t29," ", t30)
print( p2," ", p3," ", p4," ", p5," ", p6," ", p7," ", p8," ", p9," ", p10," ",p11," ", p12," ", p13," ", p14," ", p15," ", p16," ", p17," ", p18," ", p19," ", p20," ", p21," ", p22," ", p23," ", p24," ", p25," ", p26," ", p27," ", p28," ", p29," ", p30)

