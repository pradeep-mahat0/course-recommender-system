def cluster(courses_cluster_grouped,enrolled_course_ids):
    res={}
    all_id=set(courses_cluster_grouped['item'].unique())
    unknown=all_id.difference(enrolled_course_ids)

    for id in unknown:
        item_data = courses_cluster_grouped.loc[courses_cluster_grouped['item'] == id]
        # Create a dictionary for the current item and update the main dictionary
        res.update(dict(zip(item_data['item'], item_data['enrollments'])))
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)} 
    return res    

    