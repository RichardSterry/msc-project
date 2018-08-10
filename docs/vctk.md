# VCTK Dataset

 | id_vctk | age | gender | accents | region |
 | --- | --- | --- | --- | --- |
 | 253 | 22 | F | Welsh | Cardiff |

Original | <audio src="vctk/samples/p255_367.wav" controls></audio> 

![lf0](vctk/vctk_descriptive_age_etc.png)

#<img src="vctk/vctk_descriptive_age_etc.png" width="1400" />

<script>
  CsvToHtmlTable.init({
    csv_path: 'data/Health Clinics in Chicago.csv', 
    element: 'table-container', 
    allow_download: true,
    csv_options: {separator: ',', delimiter: '"'},
    datatables_options: {"paging": false}
  });
</script>