import pandas as pd
import yfinance as yf
import time
from datetime import datetime, timedelta, date
import math
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

def black_scholes_price(S_0, X, r_f, T, volatility, q):
    ''' Computes the Option Price via the Black Scholes Theorem using the given asset metrics
    
    Args:
    - S_0 (float): The Initial Price of the Asset
    - X (float): Strike Price of the Option
    - r_f (float): The Risk-Free Rate
    - T (float): Time until Option Expiration Date (in years)
    - volatility (float): The Volatility of the Asset (Std of Asset's Log Returns)
    
    Return:
    - C_0 (float): The estimated price of the option
    
    '''
    
    d_1 = (np.log(S_0/X) + T*(r_f - q + (volatility ** 2)/2)) / (volatility * np.sqrt(T))
    d_2 = (np.log(S_0/X) + T*(r_f - q - (volatility ** 2)/2)) / (volatility * np.sqrt(T))
    C_0 = S_0*np.exp(-q*T)*norm.cdf(d_1) - (X * np.exp(-r_f*T) * norm.cdf(d_2))
    
    return C_0
    

def binomial_tree_price(S_0, X, r_f, T, volatility, N, q):
    ''' Computes the Option Price via the Binomial Tree Method using the given asset metrics
    
    Args:
    - S_0 (float): The Initial Price of the Asset
    - X (float): Strike Price of the Option
    - r_f (float): The Risk-Free Rate
    - T (float): Time until Option Expiration Date (in years)
    - volatility (float): The Volatility of the Asset (Std of Asset's Log Returns)
    - N (int): The number of steps in the binomial tree
    
    Return:
    - C_0 (float): The estimated price of the option
    
    '''
    
    #Calculate the values for the up_steps, down_steps, and their probabilities for the binomial tree
    up_step = np.exp(volatility * np.sqrt(T / N))
    down_step = 1 / up_step
    p_up = (np.exp((r_f-q)*T/N) - down_step) / (up_step - down_step)
    p_down = 1 - p_up
    
    #Creates 2D Array to store future stock prices
    #Note: The upper left triangle will always remain untouched
    #asset_prices[-i - 1, j] represents the asset price with i upsteps and N+1-j downsteps
    
    #Initializes the asset prices with only the current stock price in the bottom left
    asset_prices = np.zeros((N+1, N+1))
    asset_prices[-1, 0] = S_0
    #Goes through each step of the binomial tree from left to right,
    #   calculating the subsequent upsteps and downsteps
    for j in range(0, N):
        for i in np.arange(N, -1, -1):
            if asset_prices[i, j] == 0:
                continue
            #Calculate Up Step
            asset_prices[i-1, j+1] = asset_prices[i, j] * up_step
            #Calculate Down Step
            asset_prices[i, j+1] = asset_prices[i, j] * down_step
    
    #Initializes the option values
    option_values = np.zeros((N+1, N+1))
    #Calculates the option values with the terminal stock prices for the final step of the binomial tree
    for i in range(0, N+1):
        option_values[i, -1] = max(asset_prices[i, -1] - X, 0)
    
    
    #Calculate previous option prices through backtracking
    #Note: The option value at option_values[i,j] comes from the up_step (option_values[i,j+1])
    #                                                    and the down_step (option_values[i,j])
    
    #Starts at the penultimate step, then goes back to the first step
    for j in np.arange(N-1, -1, -1):
        for i in range(1, N+1):
            #If we are currently at a position in the upper triangle
            if asset_prices[i, j] == 0:
                continue
            
            discount_term = np.exp(-1 * r_f * T / N)
            expected_option_value = p_up * option_values[i-1, j+1] + p_down * option_values[i, j+1]
            option_values[i, j] = discount_term * expected_option_value
    
    #Identifies the estimated price of the call option at the current time
    C_0 = option_values[-1, 0]
    
    return C_0
            
def monte_carlo_price(S_0, X, r_f, T, volatility, q, n_sim=10000):
    ''' Computes the Option Price via the Monte Carlo Method using the given asset metrics
    
    Args:
    - S_0 (float): The Initial Price of the Asset
    - X (float): Strike Price of the Option
    - r_f (float): The Risk-Free Rate
    - T (float): Time until Option Expiration Date (in years)
    - volatility (float): The Volatility of the Asset (Std of Asset's Log Returns)
    - n_sim (int): The number of simulations we will use
    
    Return:
    - C_0 (float): The estimated price of the option
    
    '''
    
    np.random.seed(None)
    Z_random = np.random.standard_normal(n_sim)
    exponent = T * (r_f - q - (0.5 * (volatility ** 2))) + volatility * np.sqrt(T) * Z_random
    simulated_stock_prices = S_0 * np.exp(exponent)
    payoffs = np.maximum(simulated_stock_prices - X, 0)
    discount_term = np.exp(-1 * r_f * T)
    C_0 = discount_term * np.mean(payoffs)
    return C_0