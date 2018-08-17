def foo(arg, a):
    x = 1
    y = 'xxxxxx'
    for i in range(10):
        j = 1
        k = i
    print (locals())


def columns(table):
    sql = "select lower(column_name)column_name \
    from user_tab_columns where table_name=upper('%(table)s')"  % locals()
    str = 'select from table_name = "%(table)s"' % locals()
    # str = ''.join(sql) % locals()
    # sql = 'ssssasa "%(table)s"' % {'table':table}
    print(locals())
    print(str)
    print(sql)
sss = 'question'
columns('sss')