import json
import numpy as np
import matplotlib.pyplot as mp
import statistics
import seaborn as sns


class MovieRecommendationEngine(object):
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

    def write_to_csv(self, recom_list):
        with open('recommendations.csv', 'w') as f:
            for data in recom_list:
                seq = ['Recommend ' + str(data[0]) + str(data[1])]
                f.writelines(seq)
                f.write('\n')
        f.close()

    def draw_movies_watched_counts(self, counts):
        '''{'Inception': 6, 'Pulp Fiction': 7, 'Anger Management': 4,
             'Fracture': 8, 'Serendipity': 7, 'Jerry Maguire': 7}'''
        m, r = [], []
        # 依據電影觀看次數多次至少次排序呈現
        for movie, count in sorted(counts.items(), key=lambda counts:counts[1], reverse=True):
            m.append(movie)
            r.append(count)

        mp.figure('Moive Watched Times', figsize=self.fig_size)
        sns.set(context='notebook', style='white')
        mp.title('Moive Watched Times', fontsize=self.title_fontsize)
        mp.xlabel(self.movie_name, fontsize=self.label_fontsize)
        mp.ylabel('Counts', fontsize=self.label_fontsize)
        mp.xticks(range(0, len(m)), m, rotation=10)
        mp.bar(m, r, width=0.5)
        mp.tight_layout()
        mp.show()

    def draw_user_similarity(self, user_names, max_arg, ratings):
        mp.figure(self.x, figsize=self.fig_size)
        sns.set(context='talk', style='white')

        mp.subplot(211)
        mp.title(self.x + ' '+self.y, fontsize=self.title_fontsize)
        mp.grid(":")
        mp.xlabel(self.x, fontsize=self.label_fontsize)
        mp.ylabel(self.y, fontsize=self.label_fontsize)
        mp.xticks(range(0, len(user_names)), user_names, rotation=10)
        mp.ylim(-1, len(user_names))
        mp.xlim(-0.25, len(user_names))
        mp.barh(user_names, max_arg, zorder=3)

        mp.subplot(212)
        mp.xlabel(self.y, fontsize=self.label_fontsize)
        mp.ylabel(self.x, fontsize=self.label_fontsize)
        mp.xticks(rotation=10)
        mp.yticks(range(0, len(user_names)), user_names)
        mp.xlim(-0.5, len(user_names))
        mp.scatter(user_names, max_arg, color='orange', s=150, zorder=3)
        mp.vlines(user_names, -0.2, max_arg, linestyle="dashed", color='coral')
        mp.hlines(max_arg, -0.8, user_names, linestyle="dashed", color='coral', linewidth=3)
        mp.tight_layout()
        sns.set()
        mp.show()
        self.draw_movie_rates(ratings, user_names)

    def draw_movie_rates(self, ratings, user_names):
        mp.figure('Movies Rates', figsize=self.fig_size)
        mp.title('Rates', fontsize=self.title_fontsize)
        king, item = {}, []
        for name in ratings.keys():
            user_data = ratings[name]
            king[name] = []
            for movie, score in user_data.items():
                item.append((movie, score))
                king[name].append((movie, score))

        movies, scores = [], []
        for i in king.values():
            for f in i:
                movies.append(f[0])
                scores.append(f[1])

        sns.set(style='white')
        mp.subplot(421)
        mp.title('%s' % user_names[0])
        mp.grid(axis='y')
        mp.xticks(fontsize=9, rotation=13)
        mp.yticks(self.yts, self.movie_scale)
        mp.xlim(-0.25, 5.2)
        mp.ylim(0, 5)
        mp.bar(movies[:6], scores[:6], width=0.6)

        mp.subplot(422)
        mp.title('%s' % user_names[1])
        mp.grid(axis='y')
        mp.xticks(fontsize=9, rotation=13)
        mp.xlim(-0.25, 5.2)
        mp.ylim(0, 5)
        mp.bar(movies[6:12], scores[6:12], width=0.6)

        mp.subplot(423)
        mp.title('%s' % user_names[2])
        mp.grid(axis='y')
        mp.xticks(fontsize=9, rotation=13)
        mp.yticks(self.yts, self.movie_scale)
        mp.xlim(-0.25, 5.2)
        mp.ylim(0, 5)
        mp.bar(movies[12:18], scores[12:18], width=0.6)

        mp.subplot(424)
        mp.title('%s' % user_names[3])
        mp.grid(axis='y')
        mp.xticks(fontsize=9, rotation=13)
        mp.xlim(-0.25, 5.2)
        mp.ylim(0, 5)
        mp.bar(movies[18:24], scores[18:24], width=0.6)

        mp.subplot(425)
        mp.title('%s' % user_names[4])
        mp.grid(axis='y')
        mp.ylabel('  ', fontsize=20)
        mp.xticks(fontsize=9, rotation=13)
        mp.yticks(self.yts, self.movie_scale)
        mp.xlim(-0.25, 5.2)
        mp.ylim(0, 5)
        mp.bar(movies[24:30], scores[24:30], width=0.6)

        mp.subplot(426)
        mp.title('%s' % user_names[5])
        mp.grid(axis='y')
        mp.xticks(fontsize=9, rotation=13)
        mp.xlim(-0.25, 5.2)
        mp.ylim(0, 5)
        mp.bar(movies[30:36], scores[30:36], width=0.6)

        mp.subplot(427)
        mp.title('%s' % user_names[6])
        mp.grid(axis='y')
        mp.xlabel(self.movie_name, fontsize=18)
        mp.xticks(fontsize=9, rotation=13)
        mp.yticks(self.yts, self.movie_scale)
        mp.xlim(-0.25, 5.2)
        mp.ylim(0, 5)
        mp.bar(movies[36:42], scores[36:42], width=0.6)

        mp.subplot(428)
        mp.title('%s' % user_names[7])
        mp.grid(axis='y')
        mp.xticks(fontsize=9, rotation=13)
        mp.xlabel(self.movie_name, fontsize=18)
        mp.xlim(-0.25, 5.2)
        mp.ylim(0, 5)
        mp.bar(movies[42:], scores[42:], width=0.6)

        mp.tight_layout()
        mp.show()

    def user_similarity(self, ratings, user_names, scmat):
        # 將scmat、user_names 轉類型
        scmat = np.array(scmat)
        user_names = np.array(user_names)
        # 設置一個空推薦清單列表
        recom_list = []

        '''升序排序出除了自己以外的此用者相關度'''
        # 遍歷每一位使用者，並且拿出在scmat中的排列序號
        for i, user in enumerate(user_names):
            # 將對應到的  scmat[i]得分序號索引  升序排列
            sorted_indexs = scmat[i].argsort()[::-1]
            # 須排除自己序號以外的排序，排列完後得到完成的得分升序索引
            sorted_Ascending_indexs = sorted_indexs[sorted_indexs != i]
            # 將得分升序索引放進使用者內 可得出相似度使用者升序排列
            similar_users = user_names[sorted_Ascending_indexs]
            # 再將得分升序索引放進scmat矩陣中每一行中 得出得分升序索引
            similar_scores = np.round(scmat[i, sorted_Ascending_indexs], 2) *100
            print()
            # 打印出使用者相似程度結果
            print('使用者%d: %s\n相似程度使用者:\n%s\n相似分數%%:\n%s' %
                  (i+1, user, similar_users, similar_scores))

            '''以下製作推薦清單 : 1.取相關係數 > 0(正相關) 2.取打分高 3.取權重高 '''
            # 取出大於0的正相關係數
            positive_mask = similar_scores > 0
            # 再把取出的係數　進而取出相似使用者
            similar_users = similar_users[positive_mask]
            # 再把取出的係數　進而取出相似分數
            similar_scores = similar_scores[positive_mask]

            # 建立一個空字典(score_sums)把使用者沒有看過的電影清單以權重得分作為values放進
            score_sums = {}
            # 把相似度得分放進去(weight_sums)
            user_weight_sums = {}
            # 遍歷每一位正向相似度使用者、相似度得分
            for similar_user, similar_score in zip(similar_users, similar_scores):
                # 再遍歷每一位正向相似度使用者中看過的電影、評分
                for movie, score in ratings[similar_user].items():
                    '''假設現在遍歷到'William Reynolds'。某一個正向相似度高的使用者('John')
                    看過的電影清單內中的一部Serendipity，沒有在William Reynolds看過的電影清單內，
                    就先把Serendipity打0分'''
                    if movie not in ratings[user].keys():
                        if movie not in score_sums.keys():
                            # 沒有看某部電影,　所以就會帶0　要加進去之前先打0分 後面才能計算
                            score_sums[movie] = 0
                        # += （原使用者的評分＊與自己相似度的得分＝該電影得分權重和）
                        score_sums[movie] += score * similar_score

                        if movie not in user_weight_sums.keys():
                            user_weight_sums[movie] = 0
                        user_weight_sums[movie] += similar_score

            # 建立一個空電影等級空字典，用來裝放要推薦的電影
            movie_ranks = {}
            for movie, score_sum in score_sums.items():
                movie_ranks[movie] = score_sum / user_weight_sums[movie]
            valid_recomm_index = np.array(list(movie_ranks.values())).argsort()[::-1]
            recomms = np.array(list(movie_ranks.keys()))[valid_recomm_index]
            recom_list.append((user, recomms))
        self.write_to_csv(recom_list)

    def get_all_movie_rates(self, ratings):
        # 設置電影評分空字典、電影觀看次數計算空字典
        all_movie_rates, movie_watched_counts = {}, {}
        for name in ratings.keys():
            for movie in ratings[name].keys():
                # 如果電影列表中已有此部電影名稱
                if movie in all_movie_rates:
                    # 直接加進評分
                    all_movie_rates[movie].append(ratings[name][movie])
                    # 如果看過此電影次數加一次
                    movie_watched_counts[movie] += 1
                else:  # 如果沒有此電影名稱，把名稱以及評風加進去
                    all_movie_rates[movie] = [ratings[name][movie]]
                    # 如果沒看過此電影次數直接等於一
                    movie_watched_counts[movie] = 1
        self.draw_movies_watched_counts(movie_watched_counts)
        return all_movie_rates

    def get_movie_rates_median(self, all_movie_rates, ratings):
        # 將使用者尚未看過的電影評分打上每一部電影的中位數
        mrm = {}
        for movie in all_movie_rates.keys():
            mrm[movie] = statistics.median(all_movie_rates[movie])
        for name in ratings.keys():
            for movie in set(mrm)-set(ratings[name]):
                ratings[name][movie] = mrm[movie]
        return ratings

    def get_all_movie_list(self, movie_list):
        all_movie_list = []
        count = 1
        # 打印總共電影數量、並依據英文字母排序打印
        for movie in sorted(movie_list, key=lambda movie_list :movie_list[0], reverse=False):
            all_movie_list.append(movie)
            count += 1
        return all_movie_list

    def get_user_names(self, ratings):
        user_names = list(ratings.keys())
        return user_names

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
                    # 其餘表示有看過一樣的電影
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
                score_row.append(score)
                all_movie_list.update(movies)
            scmat.append(score_row)
        sorted_all_movie_list = self.get_all_movie_list(all_movie_list)
        self.user_similarity(ratings, user_names, scmat)

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

        all_movie_rates = self.get_all_movie_rates(ratings)
        median_ratings = self.get_movie_rates_median(all_movie_rates, ratings)
        self.draw_user_similarity(user_names, max_arg, median_ratings)

    def read(self):
        with open('ratings.json', 'r') as f:
            ratings = json.loads(f.read())

        user_names = self.get_user_names(ratings)
        self.run(user_names, ratings)


if __name__ == "__main__":
    go = MovieRecommendationEngine()
    go.read()

