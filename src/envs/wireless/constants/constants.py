# --- commiunication related---------------------
#  simulation time in msec
SIM_TIME = 1000
# slot duration 
CONTENTION_WINDOW = 30 #3
# data packet time
PKT_TIME = 150
# ACK packet time
ACK_PKT_TIME = 2    #10
# the maximum number of retransmission
RETXMAX = 2
# timer for waiting for ACK NOTE: this timer starts as data transmission starts
ACK_WAIT = 250 #200
# p-csma
P = 1   #0.3 #0.05 
# the minimum required ratio of receivers in broadcast mode
Rx_R = 0.4 # 0.6          
# threshold of rssi (i.e. SINR): rss(dBm)-I_N(dBm) = rssi_theta(dB)
# I_N is the sum of interference rss power and noise power
RSSI_THETA  = 15 # 25
# energy detection threshold (). if sum_rss<ED_THETA, medium is clear
ED_THETA = -78
"""
power = 10**(dBm/10)
        power          |     dBm
-----------------------+-----------------------------
1.5848931924611143e-08 |             -78
-----------------------+-----------------------------
1e-07                  |             -70
"""


# noise in dBm
NOISE = -95
# Kref NOTE: negative
KREF = -47.6
#Nt
NT = 22
# d0 in meter
D0 = 1
# PT in dBm
PT = 20
# simulation tiral
TRIALS = 1
# ------- node ----------------------------------
# nuber of nodes
N_NODE = 5

# --------- obstacles ---------------------------
# attenuation of different materials, unit dB
GLASS = 4.5
WOOD = 2.67
CONCRETE = 12
BRICK = 3


# coordinate: up, left, width, height
OBSTACLES = [
    #[850,250,100,200],
    [0, 0, 1, 2],
    [4, 4, 1, 2],
    #[850,500,10,20],
]
# obstacle attenuation
OBSTACLES_ATT = [WOOD, GLASS]

#---- visualization ------------
# pygame screen size 
SCREEN_SIZE= [10,10]



# ---------
# nuber of nodes
N_NODE = 5
# location of each node
LOCATION = [
    [0, 0],
    [1, 1],
    [3, 3],
    [7,7],
    [2,8]
]
# message(s) lengths of each node in msec
MESSAGES= [
    [PKT_TIME],
    [PKT_TIME],
    [PKT_TIME],
    [PKT_TIME],
    [PKT_TIME]
]