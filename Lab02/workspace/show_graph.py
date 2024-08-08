import utils



utils.plot_data("SD"," AND CAST(learning_rate AS CHAR) = '0.005' AND optimizer='adamw'")
utils.plot_data("SD","AND Nu=22 AND Nt=1 AND CAST(learning_rate AS CHAR) = '0.001' AND optimizer='adamw'")