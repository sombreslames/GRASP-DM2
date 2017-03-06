<?php

namespace AppBundle\Form;

use AppBundle\Entity\Personne;
use Symfony\Component\Form\Extension\Core\Type\TextType;
use Symfony\Component\Form\AbstractType;
use Symfony\Component\Form\FormBuilderInterface;
use Symfony\Component\Form\Extension\Core\Type\ChoiceType;
use Symfony\Component\OptionsResolver\OptionsResolver;
use Symfony\Component\Form\Extension\Core\Type\DateType;
class PersonneForm extends AbstractType
{
    public function buildForm(FormBuilderInterface $builder, array $options)
    {

        $pers=array_values($options)[0];
        if($pers){
            $_nom=$pers->getNom();
            $_prenom=$pers->getPrenom();
            $_dateNaissance=$pers->getdateNaissance();
            $_sexe=$pers->getSexe();
        }
        $builder
            ->add('_nom',TextType::class)
            ->add('_prenom',TextType::class)
            ->add('_dateNaissance', DateType::class, array(
                'years' => range(date('Y') - 100, date('Y') - 18)))
            ->add('_sexe',ChoiceType::class, array(
                    'choices'  => array(
                        'Masculin' => true,
                        'Feminin' => false,
            ),))
            ->getForm();
    }

    public function configureOptions(OptionsResolver $resolver)
    {
        $resolver->setDefaults([
            'data_class' => Personne::class,
            'validation_groups' => ['Default', 'Profile']
        ]);
    }

    public function getBlockPrefix()
    {
        return 'app_bundle_personne_type';
    }
}
