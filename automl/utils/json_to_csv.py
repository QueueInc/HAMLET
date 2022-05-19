import csv


def flattenjson(b, delim):
    val = {}
    for i in b.keys():
        if isinstance(b[i], dict):
            get = flattenjson(b[i], delim)
            for j in get.keys():
                val[i + delim + j] = get[j]
        else:
            val[i] = b[i]
    return val


def json_to_csv(automl_output, args):
    to_export = automl_output["points_to_evaluate"]
    for i in range(len(to_export)):
        to_export[i][args.metric] = automl_output["evaluated_rewards"][i]

    input = [flattenjson(x, "__") for x in to_export]
    columns = [x for row in input for x in row.keys()]
    columns = list(set(columns))
    columns.sort(reverse=True)

    with open(args.output_path.replace("json", "csv"), "w") as out_file:
        csv_w = csv.writer(out_file)
        csv_w.writerow(columns)

        for i_r in input:
            csv_w.writerow(map(lambda x: i_r.get(x, ""), columns))
