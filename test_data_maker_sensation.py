# -*- coding: utf-8 -*-
import pandas as pd
input_1 = f"家族の笑顔は私の元気の源だ。いとこの集まりはいつも楽しいひととき。兄妹の絆は強く、助け合える仲。親子で過ごす時間が宝物だ。おばあちゃんの手料理はいつも優しい味。いとこの励ましで乗り越えた困難な瞬間。家族のサポートがあれば、どんなことも乗り越えられる。兄のアドバイスはいつも的確で頼りになる。両親への感謝は言葉では表しきれない。家族が一緒にいると、幸せな時間が流れる"

input_2 = f"友達との冒険は最高の思い出になる。仲間と共に笑い、涙を分かち合う。友情は心の拠り所だ。一緒にいるだけで安心感がある。仲間と共にいると、未来への希望が湧いてくる。友だちは、人生の彩りだ。仲間との時間は何よりも尊い。困難な時ほど、友達の存在が心強い。友達との絆は深く、永遠に続くものだ。仲間の応援があれば、どんな困難も乗り越えられる"

input_3 = f"愛は時折不可解で、しかし美しいものです。ふたりの心は同じリズムで響いているようだ。恋は時に冒険であり、時には穏やかな航海だ。あなたとの出会いは私の人生を豊かにしました。いつも一緒にいることが幸せなの。ときには小さなことが愛の証だ。心が通じると、言葉は不要だ。真実の愛は時を超えて続くものだ。ふたりの未来は輝かしいものになるだろう。愛することは、自分自身を見つけることでもある"

input_4 = f"人間関係がこじれると、心が重くなる。嫌なことがあると、どうしても気持ちが沈む。争いごとは心を苦しめる。他人の非難は辛いものだ。信じた仲間に裏切られると、心に深い傷が残る。悪口を言われると、なかなか忘れられない。裏切りは友情を壊す最も痛い出来事だ。嫌な感情は心を侵す毒のようだ。人を傷つけることは、自分自身も苦しめることになる。信じた人からの裏切りは心の底からくる憎しみを生む。怒りに任せて行動すると、後悔することが多い。他人との争いは心の平穏を奪う。嫉妬心が強くなると、幸福感が遠ざかる。人間関係のトラブルは、心のバランスを崩す。虚偽の言葉は心に深い疵を残す。心に積もる不満が憎しみを育む"


list = [input_1, input_2, input_3, input_4]
categories = ["親情", "友情", "愛情", "憎悪"]
list_sentence = []
list_category = []
for input, category in zip(list, categories):
    sentence_list = input.split('。')
    for sentence in sentence_list:
        list_sentence.append(sentence.replace('\n', '').replace('、', ''))
        list_category.append(category)

test_sentence = pd.DataFrame({
    'Sentence' : list_sentence,
    'Category' : list_category
})
print(test_sentence.iloc[0][0])
# print(test_sentence)
test_sentence.to_csv("D://code//FCDL//sentences_sensations.csv", index=False)

list_sentence = []
for input in list:
    sentence_list = input.split('。')
    for sentence in sentence_list:
        list_sentence.append(sentence)

for sentence in list_sentence:
    print(sentence)
print(len(list_sentence))