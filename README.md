# Movie-Recommendation-Engine--Machine-Learning
The purpose of this recommendation engine is to recommend users some un-watched movies.


## Requirements
● Python 3.8    
● json  
● numpy   
● matplotlib  
● statistics  
● seaborn


## Class
MovieRecommendationEngine


## Functions
● write_to_csv  
● draw_movies_watched_counts  
● draw_user_similarity  
● draw_movie_rates  
● user_similarity  
● get_all_movie_rates  
● get_movie_rates_median  
● get_all_movie_list  
● get_user_names  
● run  
● read   


## Create __init__
#### define all basic sets :
    def __init__(self):
            self.movie_name = 'Movie Names'
            self.x = 'Similarity'
            self.y = 'Users'
            self.movie_scale = ['V Bad', 'Bad', 'Ok', 'Good', 'V Good', 'Perfect']
            self.yts = [0, 1, 2, 3, 4, 5]
            self.fig_size = (12, 9)
            self.label_fontsize = 20
            self.title_fontsize = 25
            self.fig_size = (12, 9)
            
            
## Run codes
#### 1. Call the main finction to work :
    if __name__ == "__main__":
        go = MovieRecommendationEngine()
        go.read()

#### 2. Read json document :
    def read(self):
        with open('ratings.json', 'r') as f:
                    ratings = json.loads(f.read())

##### call get_user_names() function to get all user names
            user_names = self.get_user_names(ratings)
##### call run() function to run the engine
            self.run(user_names, ratings)
            
#### 3. Run run() function :   
    def run(self, user_names, ratings):
        all_movie_list, scmat = set(), []
        
        # 先遍歷每一行使用者
        for user_row in user_names:
            score_row = []
            
            # 每一行中去配對每一列中是否與其他人有看過相同電影
            for user_column in user_names:
                # 創建一個空的電影set，避免電影名稱重複，用來裝每一行&每一列都看過的電影集合
                movies = set()
                
                # 找尋每一行使用者看過的電影資料
                for movie in ratings[user_row]:
                    # 如果每一行使用者看過的電影也在每一列中
                    if movie in ratings[user_column]:
                        # 表示每一行與每一列都看過相同電影，加進movies 集合中
                        movies.add(movie)
                        
                # 如果電影集合長度為0
                if len(movies) == 0:
                    # 表示每一行&每一列沒有看過一樣的電影(可能看的類型迥異)，將得分設為0
                    score = 0
                # 其餘表示有看過一樣的電影，將以歐式距離來計算
                else:
                    a, b = [], []
                    for x in movies:
                        # 將每一行使用者在電影交集中的得分放進a空列表中
                        a.append(ratings[user_row][x])
                        # 將每一列使用者在電影交集中的得分放進b空列表中
                        b.append(ratings[user_column][x])
                    # 計算a,b的歐式距離
                    a, b = np.array(a), np.array(b)
                    score = 1 / (1 + np.sqrt(((a-b)**2).sum()))
                    
                # 將計算出的得分加進score_row
                score_row.append(score)
                # 將得出的電影名稱更新進all_movie_list
                all_movie_list.update(movies)
                
            scmat.append(score_row)                       
        sorted_all_movie_list = self.get_all_movie_list(all_movie_list)
        
##### call user_similarity() function to run(without median):
        self.user_similarity(ratings)
        
##### deal with other data to get similarest user:
        new_matrix, similar, max_arg = [], [], []
        for row in scmat:
            new_row = []
            for score in row:
                if score == 1.0:
                    score -= 1
                new_row.append(round(np.max(score), 2))
            new_matrix.append(new_row)

        for new_row in new_matrix:
            max_arg.append(np.argmax(new_row)+0.2)
            similar.append(user_names[np.argmax(new_row)])

##### call get_all_movie_rates() function:
        all_movie_rates = self.get_all_movie_rates(ratings)
##### call get_movie_rates_median() function to get median to put in users' unwatched score:
        median_ratings = self.get_movie_rates_median(all_movie_rates, ratings)
##### call draw_user_similarity() function to show data visualization:
        self.draw_user_similarity(user_names, max_arg, median_ratings)
![Similarity](https://user-images.githubusercontent.com/70878758/131438286-433033d4-ba07-468c-8d30-ec1f6a752a0d.jpeg)


#### 3. Write recommendations to CSV, name document'recommendation' :           
    def write_to_csv(self, recom_list):
            with open('recommendations.csv', 'w') as f:
                for data in recom_list:
                    seq = ['Recommend ' + str(data[0]) + str(data[1])]
                    f.writelines(seq)
                    f.write('\n')
            f.close()
