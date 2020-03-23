<template>
    <div>
        <div class="page-header clear-filter">
            <parallax
                    class="page-header-image"
                    style="background:linear-gradient(#8e8e8e 0%, #ffffff 80%)">
            </parallax>
            <div class="container">
            </div>
            <div class="content-center brand">
                <h1 class="h1-seo">AI 4 Covid</h1>
                <h3>AI powered solution for diagnosing COVID-19 cases based on x-ray scans.</h3>
                <n-button type="primary" @click="scroll2()" size="lg">See more</n-button>
            </div>
        </div>
        <div class="main">
            <div class="section section-images">
                <div class="container">
                    <div class="row">
                        <div class="col-md-12">
                            <div class="content-center brand">

                            <h3 class="h1-seo;text-align:center">Description </h3>
                            </div>
                            <p>
                                This project is a result of the WirVsVirus Hackathon.
                                Our objective was to assist doctors in diagnosing COVID-19 patients by training a convolutional network to discriminate between patients with and without the disease, based on an X-ray image.
                                We use a DenseNet121 pretrained on the CheXpert dataset and finetune it to the covid-chestxray-dataset.
                                Furthermore, we implemented a simple prediction service that displays the diagnosis based on an X-ray image uploaded by the user.
                            </p>
                            <br>
                            <h3 class="h1-seo;text-align:center">How it works </h3>

                            <ul>
                                <li>We first trained the COVID-19 classifier on Google Colab. See the notebook file covid_19.ipynb.</li>
                                <li>We then implemented a simple backend server in Flask, that loads the classifier and performs inference on the X-ray image uploaded by the user.</li>
                                <li>Finally, we developed a simple frontend server that allows user to upload an X-ray image and see the results of diagnosis: the predicted probability of patient having COVID-19 and the heatmap of critical regions on the X-ray that contributed to the diagnosis.</li>
                            </ul>
                            </div>

                        </div>
                    </div>
                </div>
            </div>
        </div>
</template>
<script>
    import {Parallax, Card} from '@/components';
    import Button from '../components/Button';
    import scrollTo from '../plugins/scrollMixin';

    export default {
        name: 'index',
        mixins: [scrollTo],
        bodyClass: 'index-page',
        components: {
            [Button.name]: Button,
            Parallax,
            Card,
        },
        methods: {
            scroll2() {
                this.scroll({
                    to: '.main'
                });
            }

        }

    };
</script>
<style></style>
