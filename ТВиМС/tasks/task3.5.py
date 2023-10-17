
glasnye = 'ауоыэяюеёи'
soglasnye = 'бвгджзйклмнпрстфхцчшщ'

names = ['Якунин',
         'Мартыненко',
         'Байков',
         'Гонгапшев',
         'Кудинкина',
         'Кулига',
         'Мироманов',
         'Паршаков',
         'Петренко',
         'Тумачев',
         'Хубиев',
         'Ширяева']


def run():
    glas_num = 0
    soglas_num = 0

    for name in names:
        for letter in name:
            if letter in glasnye:
                glas_num += 1
            elif letter in soglasnye:
                soglas_num += 1

    print(f'гласных букв {glas_num}\tсогласных букв {soglas_num}')



if __name__ == "__main__":
    run()