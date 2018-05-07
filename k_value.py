import mysql.connector
from mysql.connector import errorcode
from decimal import *
# import sys
# import numpy as np


def results(value,n):
    score=(n-(value))/((n*(n-1))/2)
    return score


def A_expect(Avalue,ranking, n):
    value = (n * (n - 1) / 2)
    total=0
    for x in range(n):
        total=total +(1/(1+(10**((Avalue-ranking[x][0])/1000))))

    return (total/value)


def rating(value,rank, result, a, n):
    score = value + k(a) * (result - A_expect(value,rank, n))
    return score


def Svalue(result,n):
    return (n-result)/((n*(n-1))/2)



def k(value):
    return value

def swap(ranking,y):
    temp = ranking[y][0]
    temp1 = ranking[y][1]
    temp2 = ranking[y][2]
    temp3 = ranking[y][3]
    temp4 = ranking[y][4]

    ranking[y][0] = ranking[y + 1][0]
    ranking[y][1] = ranking[y + 1][1]
    ranking[y][2] = ranking[y + 1][2]
    ranking[y][3] = ranking[y + 1][3]
    ranking[y][4] = ranking[y + 1][4]

    ranking[y + 1][0] = temp
    ranking[y + 1][1] = temp1
    ranking[y + 1][2] = temp2
    ranking[y + 1][3] = temp3
    ranking[y + 1][4] = temp4

def checkCorrection(ranking,x):
    if ranking[x][1]==1:
        return 1





a = 1


while (a < 10000):

    try:
        cnx = mysql.connector.connect(user='root',
                                      password='dereK412')
        # print ('it works')

        size = 5000  # open a 2d array
        horse_score = [[-1 for x in range(2)] for y in range(size)]  # the second index is score of horse
        jockey_score = [[-1 for x in range(2)] for y in range(size)]
        trainer_score = [[-1 for x in range(2)] for y in range(size)]

        # connect to database
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)

    else:

        run = cnx.cursor(buffered=True)
        run.execute("SELECT * FROM fyp.runs")

        i = 0  # initial value =0
        counter = 0
        score = [[-1 for x in range(6)] for y in range(500000)]
        p = 0
        ranking = [[-1 for x in range(6)] for y in range(15)]
        jockey_ranking = [[-1 for x in range(6)] for y in range(15)]
        trainer_ranking = [[-1 for x in range (6)] for y in range(15)]
        total_ranking=[[-1 for x in range(6)] for y in range(15)]
        firstTwo_ranking=[[-1 for x in range(6)] for y in range(15)]
        lastTwo_ranking=[[-1 for x in range(6)] for y in range(15)]
        firstLast_ranking=[[-1 for x in range(6)] for y in range(15)]

        winning = 0
        jockeyWinning = 0
        tranerWinning = 0
        firstTwo_winning=0
        lastTwo_winning=0
        firstLast_winning=0
        total_winning=0

        tempCal=0
        tempCal1=0
        n = 0





        for row in run.fetchall():
            num_winnning=0
            valueOfHorse = valueOfTrainer = valueOfJockey = 1500
            counter = counter + 1

            if horse_score[row[2]][1] != -1:  # assign value to corresponding horse
                valueOfHorse = round(horse_score[row[2]][1],2)
            elif horse_score[row[2]][1] == -1:
                valueOfHorse = 1500

            if jockey_score[row[36]][1] != -1:  # assign value to jockey
                valueOfJockey = round(jockey_score[row[36]][1],2)
            elif jockey_score[row[36]][1] == -1:
                valueOfJockey = 1500

            if trainer_score[row[35]][1] != -1:  # assign value to trainer
                valueOfTrainer = round(trainer_score[row[35]][1],2)
            elif trainer_score[row[35]][1] == -1:
                valueOfTrainer = 1500

            if p == row[0]:
                if row[3]==1:
                    num_winnning=num_winnning+1

                ranking[row[1]][0] = int(valueOfHorse)
                ranking[row[1]][1] = row[3]
                ranking[row[1]][2] = row[33]
                ranking[row[1]][3] = row[34]
                ranking[row[1]][4]=row[2]

                jockey_ranking[row[1]][0] = int(valueOfJockey)
                jockey_ranking[row[1]][1] = row[3]
                jockey_ranking[row[1]][2] = row[33]
                jockey_ranking[row[1]][3] = row[34]
                jockey_ranking[row[1]][4]= row[36]

                trainer_ranking[row[1]][0] = int(valueOfTrainer)
                trainer_ranking[row[1]][1] = row[3]
                trainer_ranking[row[1]][2] = row[33]
                trainer_ranking[row[1]][3] = row[34]
                trainer_ranking[row[1]][4]=row[35]

                firstTwo_ranking[row[1]][0]= int(valueOfHorse+valueOfJockey)
                firstTwo_ranking[row[1]][1] = row[3]
                firstTwo_ranking[row[1]][2] = row[33]
                firstTwo_ranking[row[1]][3] = row[34]

                firstLast_ranking[row[1]][0]=int(valueOfHorse+valueOfTrainer)
                firstLast_ranking[row[1]][1] = row[3]
                firstLast_ranking[row[1]][2] = row[33]
                firstLast_ranking[row[1]][3] = row[34]

                lastTwo_ranking[row[1]][0]=int(trainer_ranking[row[1]][0]+ jockey_ranking[row[1]][0])
                lastTwo_ranking[row[1]][1] = row[3]
                lastTwo_ranking[row[1]][2] = row[33]
                lastTwo_ranking[row[1]][3] = row[34]

                total_ranking[row[1]][0]= int (valueOfJockey+valueOfTrainer+valueOfHorse)
                total_ranking[row[1]][1] = row[3]
                total_ranking[row[1]][2] = row[33]
                total_ranking[row[1]][3] = row[34]

                n = n + 1
            else:
                for x in range(len(ranking) - 1, 0, -1):
                    for y in range(x):
                        if ranking[y][0] < ranking[y + 1][0]:
                            swap(ranking,y)

                        if jockey_ranking[y][0] < jockey_ranking[y + 1][0]:
                            swap(jockey_ranking,y)

                        if trainer_ranking[y][0] < trainer_ranking[y + 1][0]:
                            swap(trainer_ranking,y)

                        if firstTwo_ranking[y][0]<firstTwo_ranking[y+1][0]:
                            swap(firstTwo_ranking,y)

                        if lastTwo_ranking[y][0]<lastTwo_ranking[y+1][0]:
                            swap(lastTwo_ranking,y)

                        if firstLast_ranking[y][0]<firstLast_ranking[y+1][0]:
                            swap(firstLast_ranking,y)

                        if total_ranking[y][0]<total_ranking[y+1][0]:
                            swap(total_ranking,y)

                correct = 0
                jockey_correct = 0
                trainer_correct = 0
                firstTwo_correct=0
                lastTwo_correct=0
                firstLast_correct=0
                total=0



                if row[0]>1000 and row[0]<2000:
                    for x in range(1):
                        correct=checkCorrection(ranking,x)
                        jockey_correct=checkCorrection(jockey_ranking,x)
                        trainer_correct=checkCorrection(trainer_ranking,x)
                        firstTwo_correct=checkCorrection(firstTwo_ranking,x)
                        firstLast_correct=checkCorrection(firstLast_ranking,x)
                        total=checkCorrection(total_ranking,x)
                        lastTwo_correct=checkCorrection(lastTwo_ranking,x)



                        if correct ==1:
                            winning = winning + 1

                        if jockey_correct ==1:
                            jockeyWinning = jockeyWinning + 1


                        if trainer_correct ==1:
                            tranerWinning = tranerWinning + 1


                        if firstLast_correct==1:
                            firstLast_winning=firstLast_winning+1

                        if firstTwo_correct==1:
                            firstTwo_winning=firstTwo_winning+1

                        if total==1:
                            total_winning=total_winning+1

                        if lastTwo_correct==1:
                            lastTwo_winning=lastTwo_winning+1



                    for x in range(len(jockey_ranking)):
                        getcontext().prec = 6

                        if num_winnning==1:

                            # a horse win a race
                            jockey_score[jockey_ranking[x][4]][1] = int (rating(valueOfJockey,jockey_ranking, results(row[3],n), a, n))
                            horse_score[ranking[x][4]][1] = rating(valueOfHorse,ranking, results(row[3],n), a, n)
                            trainer_score[trainer_ranking[x][4]][1] = int (rating(valueOfTrainer,trainer_ranking, results(row[3],n), a, n))
                        else:

                            # double horses win a race
                            jockey_score[jockey_ranking[x][4]][1] = int (rating(valueOfJockey,jockey_ranking, results(row[3],n)+0.5, a, n-1))
                            horse_score[ranking[x][4]][1] = rating(valueOfHorse,ranking, results(row[3],n)+0.5, a, n-1)
                            trainer_score[trainer_ranking[x][4]][1] = int (rating(valueOfTrainer,trainer_ranking, results(row[3],n)+0.5, a, n-1))


                    n = 0
                    p = row[0]
                    ranking = [[-1 for x in range(6)] for y in range(15)]
                    jockey_ranking=[[-1 for x in range(6)] for y in range(15)]
                    trainer_ranking=[[-1 for x in range(6)] for y in range(15)]
                    total_ranking = [[-1 for x in range(6)] for y in range(15)]
                    firstTwo_ranking = [[-1 for x in range(6)] for y in range(15)]
                    lastTwo_ranking = [[-1 for x in range(6)] for y in range(15)]
                    firstLast_ranking = [[-1 for x in range(6)] for y in range(15)]


                    # consider combination of those three factors
        print(winning," ",jockeyWinning," ",tranerWinning," ",firstTwo_winning," ",lastTwo_winning," ",firstLast_winning," ",total_winning," ",a)

        cnx.close()
        a = a + 1
