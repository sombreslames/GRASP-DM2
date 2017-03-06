<?php
/**
 * Created by PhpStorm.
 * User: jack
 * Date: 04/03/17
 * Time: 18:54
 */

namespace AppBundle\Repository;
use Doctrine\ORM\EntityRepository;
use Doctrine\ORM\QueryBuilder;
class PersonneRepository extends EntityRepository
{
    public function loadProfile($id)
    {
        return $this->createQueryBuilder('p')
            ->where('p.ID_personne = :id')
            ->setParameter('id', $id)
            ->getQuery()
            ->getOneOrNullResult();
    }

}