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
        scmat = np.array(scmat)
        user_names = np.array(user_names)
        recom_list = []
        
        for i, user in enumerate(user_names):
            sorted_indexs = scmat[i].argsort()[::-1]
            sorted_Ascending_indexs = sorted_indexs[sorted_indexs != i]
            similar_users = user_names[sorted_Ascending_indexs]
            similar_scores = np.round(scmat[i, sorted_Ascending_indexs], 2) *100
            print()
            print('使用者%d: %s\n相似程度使用者:\n%s\n相似分數%%:\n%s' %
                  (i+1, user, similar_users, similar_scores))

            positive_mask = similar_scores > 0
            similar_users = similar_users[positive_mask]
            similar_scores = similar_scores[positive_mask]
            
            score_sums, user_weight_sums = {}, {}
            for similar_user, similar_score in zip(similar_users, similar_scores):
                for movie, score in ratings[similar_user].items():
                    if movie not in ratings[user].keys():
                        if movie not in score_sums.keys():
                            score_sums[movie] = 0
                        score_sums[movie] += score * similar_score

                        if movie not in user_weight_sums.keys():
                            user_weight_sums[movie] = 0
                        user_weight_sums[movie] += similar_score

            movie_ranks = {}
            for movie, score_sum in score_sums.items():
                movie_ranks[movie] = score_sum / user_weight_sums[movie]
            valid_recomm_index = np.array(list(movie_ranks.values())).argsort()[::-1]
            recomms = np.array(list(movie_ranks.keys()))[valid_recomm_index]
            recom_list.append((user, recomms))
        self.write_to_csv(recom_list)

    def get_all_movie_rates(self, ratings):
        all_movie_rates, movie_watched_counts = {}, {}
        for name in ratings.keys():
            for movie in ratings[name].keys():
                if movie in all_movie_rates:
                    all_movie_rates[movie].append(ratings[name][movie])
                    movie_watched_counts[movie] += 1
                else: 
                    all_movie_rates[movie] = [ratings[name][movie]]
                    movie_watched_counts[movie] = 1
        self.draw_movies_watched_counts(movie_watched_counts)
        return all_movie_rates

    def get_movie_rates_median(self, all_movie_rates, ratings):
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
            for user_column in user_names:
                movies = set()
                for movie in ratings[user_row]:
                    if movie in ratings[user_column]:
                        movies.add(movie)
                if len(movies) == 0:
                    score = 0
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
